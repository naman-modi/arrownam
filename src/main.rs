use arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, BooleanBuilder, GenericListArray, Int32Builder,
    Int64Builder, ListBuilder, StringArray, StringBuilder, TimestampMillisecondArray,
};
use arrow::datatypes::{DataType, Field, Int32Type, Int64Type, Schema, TimeUnit};
use arrow::error::{ArrowError, Result as ArrowResult};
use arrow::ipc::writer::FileWriter;
use arrow::record_batch::RecordBatch;
use arrow_buffer::Buffer;
use arrow_ipc::convert::fb_to_schema;
use arrow_ipc::reader::{read_footer_length, FileDecoder};
use arrow_ipc::{root_as_footer, Block};
use bytes::Bytes;
use chrono::Utc;
use memmap2::Mmap;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::File;
use std::process;
use std::sync::Arc;
use std::time::{Duration, Instant};
use uuid::Uuid;

/// Structure to hold benchmark metrics for queries
#[derive(Debug, Clone)]
pub struct QueryStats {
    pub query_type: String,
    pub field_name: String,
    pub doc_ids_count: usize,
    pub result_rows: usize,
    pub setup_time_ms: u128,
    pub process_time_ms: u128,
    pub total_time_ms: u128,
    pub memory_before_mb: f64,
    pub memory_after_mb: f64,
    pub memory_impact_mb: f64,
}

impl QueryStats {
    pub fn new(query_type: &str, field_name: &str, doc_ids_count: usize) -> Self {
        let (rss, _vsz) = get_memory_usage_stats();
        QueryStats {
            query_type: query_type.to_string(),
            field_name: field_name.to_string(),
            doc_ids_count,
            result_rows: 0,
            setup_time_ms: 0,
            process_time_ms: 0,
            total_time_ms: 0,
            memory_before_mb: rss as f64 / 1024.0 / 1024.0,
            memory_after_mb: 0.0,
            memory_impact_mb: 0.0,
        }
    }

    pub fn finish(mut self, result_rows: usize) -> Self {
        let (rss, _vsz) = get_memory_usage_stats();
        self.result_rows = result_rows;
        self.memory_after_mb = rss as f64 / 1024.0 / 1024.0;
        self.memory_impact_mb = self.memory_after_mb - self.memory_before_mb;
        self
    }

    pub fn add_timing(&mut self, setup_time: Duration, process_time: Duration) {
        self.setup_time_ms = setup_time.as_millis();
        self.process_time_ms = process_time.as_millis();
        self.total_time_ms = setup_time.as_millis() + process_time.as_millis();
    }

    pub fn print_benchmark(&self) {
        println!("\n=== Query Performance Metrics ===");
        println!("Query Type: {}", self.query_type);
        println!("Field Name: {}", self.field_name);
        println!("Doc IDs Count: {}", self.doc_ids_count);
        println!("Result Rows: {}", self.result_rows);
        println!("Setup Time: {:.2} ms", self.setup_time_ms);
        println!("Process Time: {:.2} ms", self.process_time_ms);
        println!(
            "Total Time: {:.2} ms ({:.2} s)",
            self.total_time_ms,
            self.total_time_ms as f64 / 1000.0
        );
        println!("Memory Before: {:.2} MB", self.memory_before_mb);
        println!("Memory After: {:.2} MB", self.memory_after_mb);
        println!("Memory Impact: {:.2} MB", self.memory_impact_mb);
        println!("===============================");
    }
}

/// Measure setup and processing time for a query
pub fn measure_query_timing<T, F1, F2>(setup_fn: F1, process_fn: F2) -> (T, Duration, Duration)
where
    F1: FnOnce() -> T,
    F2: FnOnce(T) -> T,
{
    // Measure setup time
    let setup_start = Instant::now();
    let result = setup_fn();
    let setup_time = setup_start.elapsed();

    // Measure processing time
    let process_start = Instant::now();
    let processed_result = process_fn(result);
    let process_time = process_start.elapsed();

    (processed_result, setup_time, process_time)
}

/// Print a benchmark summary table from multiple query stats
pub fn print_benchmark_table(all_stats: &[QueryStats]) {
    println!("\n======================= ARROW BENCHMARK SUMMARY =======================");
    println!(
        "| {:<30} | {:<20} | {:<8} | {:<11} | {:<10} | {:<13} |",
        "Query Type", "Field", "Doc IDs", "Result Rows", "Total Time", "Memory Impact"
    );
    println!("|--------------------------------|----------------------|----------|-------------|------------|---------------|");

    for stat in all_stats {
        println!(
            "| {:<30} | {:<20} | {:<8} | {:<11} | {:<8.2}s | {:<13.2} |",
            stat.query_type,
            stat.field_name,
            stat.doc_ids_count,
            stat.result_rows,
            stat.total_time_ms as f64 / 1000.0,
            stat.memory_impact_mb
        );
    }
    println!("=====================================================================");
}

/// Incrementally decodes RecordBatches from an IPC file stored in an Arrow
/// Buffer using the FileDecoder API.
struct IPCBufferDecoder {
    /// Memory (or memory mapped) Buffer with the data
    buffer: Buffer,
    /// Decoder that reads Arrays that refers to the underlying buffers
    decoder: FileDecoder,
    /// Location of the batches within the buffer
    batches: Vec<Block>,
}

impl IPCBufferDecoder {
    fn new(buffer: Buffer) -> Self {
        let trailer_start = buffer.len() - 10;
        let footer_len = read_footer_length(buffer[trailer_start..].try_into().unwrap()).unwrap();
        let footer = root_as_footer(&buffer[trailer_start - footer_len..trailer_start]).unwrap();
        let schema = fb_to_schema(footer.schema().unwrap());
        let mut decoder = FileDecoder::new(Arc::new(schema), footer.version());

        // Read dictionaries
        for block in footer.dictionaries().iter().flatten() {
            let block_len = block.bodyLength() as usize + block.metaDataLength() as usize;
            let data = buffer.slice_with_length(block.offset() as _, block_len);
            decoder.read_dictionary(block, &data).unwrap();
        }

        // convert to Vec from the flatbuffers Vector to avoid having a direct dependency on flatbuffers
        let batches = footer
            .recordBatches()
            .map(|b| b.iter().copied().collect())
            .unwrap_or_default();

        Self {
            buffer,
            decoder,
            batches,
        }
    }

    /// Return the number of RecordBatches in this buffer
    fn num_batches(&self) -> usize {
        self.batches.len()
    }

    /// Return the RecordBatch at message index `i`.
    ///
    /// This may return None if the IPC message was None
    fn get_batch(&self, i: usize) -> ArrowResult<Option<RecordBatch>> {
        let block = &self.batches[i];
        let block_len = block.bodyLength() as usize + block.metaDataLength() as usize;
        let data = self
            .buffer
            .slice_with_length(block.offset() as _, block_len);
        self.decoder.read_record_batch(block, &data)
    }
}

/// Memory usage statistics
fn get_memory_usage_stats() -> (u64, u64) {
    let pid = process::id();
    // Get RSS (Resident Set Size) and VSZ (Virtual Memory Size)
    if let Ok(output) = std::process::Command::new("ps")
        .args(["-o", "rss=,vsz=", "-p", &pid.to_string()])
        .output()
    {
        if let Ok(stats_str) = String::from_utf8(output.stdout) {
            let stats: Vec<&str> = stats_str.split_whitespace().collect();
            if stats.len() >= 2 {
                if let (Ok(rss), Ok(vsz)) = (stats[0].parse::<u64>(), stats[1].parse::<u64>()) {
                    return (rss * 1024, vsz * 1024); // Convert KB to bytes
                }
            }
        }
    }
    (0, 0)
}

/// Print memory usage statistics
fn print_memory_stats(stage: &str) {
    let (rss, vsz) = get_memory_usage_stats();
    println!("\n=== Memory Stats at {} ===", stage);
    println!(
        "RSS (Physical Memory): {:.2} MB",
        rss as f64 / 1024.0 / 1024.0
    );
    println!(
        "VSZ (Virtual Memory): {:.2} MB",
        vsz as f64 / 1024.0 / 1024.0
    );
    println!("==============================\n");
}

/// Write a record batch to an Arrow IPC file
fn write_batch_ipc(path: &str, batch: &RecordBatch) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = FileWriter::try_new(file, &batch.schema()).unwrap();
    writer.write(batch).unwrap();
    writer.finish().unwrap();
    Ok(())
}

/// Creates a schema for log records with nested fields
/// The schema supports the format from generate_random_json:
/// - doc_id: document ID for lookups
/// - timestamp: millisecond timestamp
/// - level: log level (info, warn, error, debug)
/// - message: log message
/// - source: object with ip, host, region
/// - user: object with id, session_id, metrics (login_time_ms, clicks, active)
/// - payload_size: size of payload
/// - tags: array of strings
/// - answers: array of objects with nxDomain boolean
fn create_logs_schema() -> Schema {
    let doc_id_field = Field::new("doc_id", DataType::Int64, false);

    let timestamp_field = Field::new(
        "timestamp",
        DataType::Timestamp(TimeUnit::Millisecond, None),
        false,
    );

    let level_field = Field::new("level", DataType::Utf8, false);

    let message_field = Field::new("message", DataType::Utf8, false);

    // Source fields
    let source_ip_field = Field::new("source.ip", DataType::Utf8, false);
    let source_host_field = Field::new("source.host", DataType::Utf8, false);
    let source_region_field = Field::new("source.region", DataType::Utf8, false);

    // User fields
    let user_id_field = Field::new("user.id", DataType::Utf8, false);
    let user_session_id_field = Field::new("user.session_id", DataType::Utf8, false);
    let user_metrics_login_time_field =
        Field::new("user.metrics.login_time_ms", DataType::Int32, false);
    let user_metrics_clicks_field = Field::new("user.metrics.clicks", DataType::Int32, false);
    let user_metrics_active_field = Field::new("user.metrics.active", DataType::Boolean, false);

    let payload_size_field = Field::new("payload_size", DataType::Int32, false);

    // Tags as list of strings
    let tags_field = Field::new(
        "tags",
        DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
        true,
    );

    // Answers as list of structs with nxDomain
    let answers_nx_domain_field = Field::new(
        "answers.nxDomain",
        DataType::List(Arc::new(Field::new("item", DataType::Boolean, true))),
        true,
    );

    Schema::new(vec![
        doc_id_field,
        timestamp_field,
        level_field,
        message_field,
        source_ip_field,
        source_host_field,
        source_region_field,
        user_id_field,
        user_session_id_field,
        user_metrics_login_time_field,
        user_metrics_clicks_field,
        user_metrics_active_field,
        payload_size_field,
        tags_field,
        answers_nx_domain_field,
    ])
}

/// Creates a large record batch with specified number of rows
fn create_large_record_batch(num_rows: usize) -> ArrowResult<RecordBatch> {
    let schema = create_logs_schema();

    // Create builders with larger capacity
    let mut doc_id_builder = Int64Builder::new();
    let mut timestamp_builder = TimestampMillisecondArray::builder(num_rows);
    let mut level_builder = StringBuilder::new();
    let mut message_builder = StringBuilder::new();
    let mut source_ip_builder = StringBuilder::new();
    let mut source_host_builder = StringBuilder::new();
    let mut source_region_builder = StringBuilder::new();
    let mut user_id_builder = StringBuilder::new();
    let mut user_session_id_builder = StringBuilder::new();
    let mut user_metrics_login_time_builder = Int32Builder::new();
    let mut user_metrics_clicks_builder = Int32Builder::new();
    let mut user_metrics_active_builder = BooleanArray::builder(num_rows);
    let mut payload_size_builder = Int32Builder::new();
    let mut tags_builder = ListBuilder::new(StringBuilder::new());
    let mut answers_nx_domain_builder = ListBuilder::new(BooleanBuilder::new());

    // For tracking raw data size
    let mut raw_string_size = 0;
    let mut raw_int_size = 0;
    let mut total_strings = 0;

    // Generate large amount of data using the random JSON generator
    for i in 0..num_rows {
        // Generate a random JSON document for this record
        let json_data = generate_random_json(i);

        // Extract values from the JSON and add to builders
        doc_id_builder.append_value(i as i64);
        raw_int_size += std::mem::size_of::<i64>();

        // Make sure we're using timestamp in milliseconds to match the schema
        let timestamp_millis = json_data["timestamp"].as_i64().unwrap();
        println!("Adding timestamp: {}", timestamp_millis); // Debugging
        timestamp_builder.append_value(timestamp_millis);
        raw_int_size += std::mem::size_of::<i64>();

        let level = json_data["level"].as_str().unwrap();
        level_builder.append_value(level);
        raw_string_size += level.len();

        let message = json_data["message"].as_str().unwrap();
        message_builder.append_value(message);
        raw_string_size += message.len();

        let ip = json_data["source"]["ip"].as_str().unwrap();
        source_ip_builder.append_value(ip);
        raw_string_size += ip.len();

        let host = json_data["source"]["host"].as_str().unwrap();
        source_host_builder.append_value(host);
        raw_string_size += host.len();

        let region = json_data["source"]["region"].as_str().unwrap();
        source_region_builder.append_value(region);
        raw_string_size += region.len();

        let user_id = json_data["user"]["id"].as_str().unwrap();
        user_id_builder.append_value(user_id);
        raw_string_size += user_id.len();

        let session_id = json_data["user"]["session_id"].as_str().unwrap();
        user_session_id_builder.append_value(session_id);
        raw_string_size += session_id.len();

        let login_time = json_data["user"]["metrics"]["login_time_ms"]
            .as_i64()
            .unwrap() as i32;
        user_metrics_login_time_builder.append_value(login_time);
        raw_int_size += std::mem::size_of::<i32>();

        let clicks = json_data["user"]["metrics"]["clicks"].as_i64().unwrap() as i32;
        user_metrics_clicks_builder.append_value(clicks);
        raw_int_size += std::mem::size_of::<i32>();

        let active = json_data["user"]["metrics"]["active"].as_bool().unwrap();
        user_metrics_active_builder.append_value(active);
        raw_int_size += 1; // 1 byte for boolean

        let payload_size = json_data["payload_size"].as_i64().unwrap() as i32;
        payload_size_builder.append_value(payload_size);
        raw_int_size += std::mem::size_of::<i32>();

        // Handle tags array
        if let Some(tags_array) = json_data["tags"].as_array() {
            for tag in tags_array {
                let tag_str = tag.as_str().unwrap();
                tags_builder.values().append_value(tag_str);
                raw_string_size += tag_str.len();
                total_strings += 1;
            }
        }
        tags_builder.append(true);

        // Handle answers array with nxDomain
        if let Some(answers) = json_data["answers"].as_array() {
            for answer in answers {
                let nx_domain = answer["nxDomain"].as_bool().unwrap();
                answers_nx_domain_builder.values().append_value(nx_domain);
                raw_int_size += 1; // 1 byte for boolean
            }
        }
        answers_nx_domain_builder.append(true);
    }

    // Calculate and print raw data size statistics
    let total_raw_size = raw_string_size + raw_int_size;
    println!("\n=== Raw Data Size Statistics ===");
    println!("Number of rows: {}", num_rows);
    println!("Number of strings: {}", total_strings);
    println!(
        "Raw string data size: {} bytes ({:.2} MB)",
        raw_string_size,
        raw_string_size as f64 / (1024.0 * 1024.0)
    );
    println!(
        "Raw integer data size: {} bytes ({:.2} MB)",
        raw_int_size,
        raw_int_size as f64 / (1024.0 * 1024.0)
    );
    println!(
        "Total raw data size: {} bytes ({:.2} MB)",
        total_raw_size,
        total_raw_size as f64 / (1024.0 * 1024.0)
    );
    println!("Average per row: {} bytes", total_raw_size / num_rows);
    println!("==============================\n");

    let arrays: Vec<ArrayRef> = vec![
        Arc::new(doc_id_builder.finish()),
        Arc::new(timestamp_builder.finish()),
        Arc::new(level_builder.finish()),
        Arc::new(message_builder.finish()),
        Arc::new(source_ip_builder.finish()),
        Arc::new(source_host_builder.finish()),
        Arc::new(source_region_builder.finish()),
        Arc::new(user_id_builder.finish()),
        Arc::new(user_session_id_builder.finish()),
        Arc::new(user_metrics_login_time_builder.finish()),
        Arc::new(user_metrics_clicks_builder.finish()),
        Arc::new(user_metrics_active_builder.finish()),
        Arc::new(payload_size_builder.finish()),
        Arc::new(tags_builder.finish()),
        Arc::new(answers_nx_domain_builder.finish()),
    ];

    RecordBatch::try_new(Arc::new(schema), arrays)
}

/// Generate a random JSON document according to the specified schema
fn generate_random_json(i: usize) -> Value {
    let mut rng = rand::thread_rng();
    let levels = ["info", "warn", "error", "debug"];
    let regions = ["us-east-1", "eu-west-2", "ap-south-1"];

    let answers_len = rng.gen_range(0..=5);
    let answers = (0..answers_len)
        .map(|_| {
            json!({
                "nxDomain": rng.gen_bool(0.5),
            })
        })
        .collect::<Vec<_>>();

    // Create current timestamp in milliseconds
    let current_time = Utc::now().timestamp_millis();

    json!({
        "doc_id": i, // Add a document ID for efficient lookups
        "timestamp": current_time,
        "level": levels[rng.gen_range(0..levels.len())],
        "message": format!("Log message for record {}", i),
        "source": {
            "ip": format!("192.168.{}.{}", rng.gen_range(1..255), rng.gen_range(1..255)),
            "host": format!("server-{}.local", rng.gen_range(1..100)),
            "region": regions[rng.gen_range(0..regions.len())],
        },
        "user": {
            "id": format!("user_{}", rng.gen_range(1000..10000)),
            "session_id": Uuid::new_v4().to_string(),
            "metrics": {
                "login_time_ms": rng.gen_range(10..500),
                "clicks": rng.gen_range(0..50),
                "active": rng.gen_bool(0.8),
            }
        },
        "payload_size": rng.gen_range(100..10_240),
        "tags": (0..rng.gen_range(1..6))
            .map(|i| format!("tag_{}", i * rng.gen_range(1..10)))
            .collect::<Vec<_>>(),
        "answers": answers
    })
}

/// Process a batch of list array values for specific document IDs using a filter bitset
fn process_list_array_batch_with_filter(
    array: &ArrayRef,
    filter: &[bool],
    doc_ids: &[usize],
    _indices: &[usize],
    _all_doc_ids: &[usize],
    value_to_doc_ids: &mut HashMap<String, Vec<usize>>,
) -> Result<(), ArrowError> {
    match array.data_type() {
        DataType::List(field) => {
            match field.data_type() {
                DataType::Utf8 => {
                    let list_array: &GenericListArray<i32> =
                        array.as_any().downcast_ref().ok_or_else(|| {
                            ArrowError::InvalidArgumentError(
                                "Failed to downcast to list array".to_string(),
                            )
                        })?;

                    // Process only the filtered document IDs
                    for &doc_id in doc_ids {
                        // Only process if the document is in our filter
                        if filter[doc_id] && !list_array.is_null(doc_id) {
                            let original_doc_id = doc_id; // We're using the actual doc ID now
                            let values = list_array.value(doc_id);
                            let string_array = values
                                .as_any()
                                .downcast_ref::<StringArray>()
                                .ok_or_else(|| {
                                    ArrowError::InvalidArgumentError(
                                        "Failed to downcast to string array".to_string(),
                                    )
                                })?;

                            for j in 0..string_array.len() {
                                if !string_array.is_null(j) {
                                    let value = string_array.value(j).to_string();
                                    value_to_doc_ids
                                        .entry(value)
                                        .or_default()
                                        .push(original_doc_id);
                                }
                            }
                        }
                    }
                }
                DataType::Int32 => {
                    let list_array: &GenericListArray<i32> =
                        array.as_any().downcast_ref().ok_or_else(|| {
                            ArrowError::InvalidArgumentError(
                                "Failed to downcast to list array".to_string(),
                            )
                        })?;

                    // Process only the filtered document IDs
                    for &doc_id in doc_ids {
                        // Only process if the document is in our filter
                        if filter[doc_id] && !list_array.is_null(doc_id) {
                            let original_doc_id = doc_id; // We're using the actual doc ID now
                            let values = list_array.value(doc_id);
                            let int_array = values.as_primitive::<Int32Type>();

                            for j in 0..int_array.len() {
                                if !int_array.is_null(j) {
                                    let value = int_array.value(j).to_string();
                                    value_to_doc_ids
                                        .entry(value)
                                        .or_default()
                                        .push(original_doc_id);
                                }
                            }
                        }
                    }
                }
                _ => {
                    // For other types, use the generic approach
                    if let Some(list_array) = array.as_any().downcast_ref::<GenericListArray<i32>>()
                    {
                        for &doc_id in doc_ids {
                            // Only process if the document is in our filter
                            if filter[doc_id] && !list_array.is_null(doc_id) {
                                let original_doc_id = doc_id; // We're using the actual doc ID now
                                let values = list_array.value(doc_id);
                                let len = values.len();

                                for j in 0..len {
                                    if !values.is_null(j) {
                                        let value = format!("{:?}", values.as_any());
                                        value_to_doc_ids
                                            .entry(value)
                                            .or_default()
                                            .push(original_doc_id);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        _ => {
            return Err(ArrowError::InvalidArgumentError(
                "Expected a list array".to_string(),
            ))
        }
    }

    Ok(())
}

/// Process a batch of scalar array values for specific document IDs using a filter bitset
fn process_scalar_array_batch_with_filter(
    array: &ArrayRef,
    filter: &[bool],
    doc_ids: &[usize],
    _indices: &[usize],
    _all_doc_ids: &[usize],
    value_to_doc_ids: &mut HashMap<String, Vec<usize>>,
) -> Result<(), ArrowError> {
    // Process only the filtered document IDs
    for &doc_id in doc_ids {
        // Only process if the document is in our filter
        if filter[doc_id] && !array.is_null(doc_id) {
            let original_doc_id = doc_id; // We're using the actual doc ID now
            let value = match array.data_type() {
                DataType::Utf8 => {
                    let string_array = array.as_any().downcast_ref::<StringArray>().unwrap();
                    string_array.value(doc_id).to_string()
                }
                DataType::Boolean => {
                    let bool_array = array.as_boolean();
                    bool_array.value(doc_id).to_string()
                }
                DataType::Int32 => {
                    let int_array = array.as_primitive::<Int32Type>();
                    int_array.value(doc_id).to_string()
                }
                DataType::Int64 => {
                    let int_array = array.as_primitive::<Int64Type>();
                    int_array.value(doc_id).to_string()
                }
                _ => format!("{:?}", array.as_any()),
            };

            value_to_doc_ids
                .entry(value)
                .or_default()
                .push(original_doc_id);
        }
    }

    Ok(())
}

/// Retrieves field values for all documents in an Arrow IPC file using zero-copy approach
/// This implementation processes all documents in the file with minimal memory usage
/// by using the FileDecoder API with memory-mapped files
fn get_field_values_zero_copy(
    field_name: &str,
    file_path: &str,
) -> Result<(HashMap<String, Vec<usize>>, QueryStats), ArrowError> {
    // Create stats object
    let mut stats = QueryStats::new("get_field_values_zero_copy", field_name, 0);

    // Reserve a large initial capacity for the HashMap to avoid frequent resizing
    let mut value_to_doc_ids: HashMap<String, Vec<usize>> = HashMap::with_capacity(1_000_000);

    // Measure setup time
    let setup_start = Instant::now();

    // Open the file and memory map it
    let file = File::open(file_path)
        .map_err(|e| ArrowError::InvalidArgumentError(format!("Failed to open file: {}", e)))?;

    // Get file metadata
    let file_size = file
        .metadata()
        .map_err(|e| {
            ArrowError::InvalidArgumentError(format!("Failed to get file metadata: {}", e))
        })?
        .len();

    println!("File size: {} bytes", file_size);

    // Memory mapping allows us to access file contents without loading entire file into memory
    let mmap = unsafe {
        Mmap::map(&file).map_err(|e| {
            ArrowError::InvalidArgumentError(format!("Failed to memory map file: {}", e))
        })?
    };

    // Use the zero-copy approach
    // We need to copy from the mmap to handle lifetime issues
    let bytes = Bytes::copy_from_slice(&mmap[..]);
    // Create Buffer from Bytes (zero-copy)
    let buffer = Buffer::from(bytes);

    // Create the IPCBufferDecoder which will efficiently decode record batches
    let decoder = IPCBufferDecoder::new(buffer);
    println!("Number of batches in file: {}", decoder.num_batches());

    let setup_time = setup_start.elapsed();

    // Measure processing time
    let process_start = Instant::now();

    // We need to get the first batch to access its schema
    if let Ok(Some(first_batch)) = decoder.get_batch(0) {
        let schema = first_batch.schema();
        let field_index = schema.index_of(field_name).map_err(|_| {
            ArrowError::InvalidArgumentError(format!("Field '{}' not found in schema", field_name))
        })?;

        // Process batches
        let mut batch_count = 0;
        let mut total_rows = 0;
        let mut strings_seen = 0;

        // Process all batches one by one
        for batch_idx in 0..decoder.num_batches() {
            match decoder.get_batch(batch_idx) {
                Ok(Some(batch)) => {
                    batch_count += 1;
                    let batch_rows = batch.num_rows();
                    total_rows += batch_rows;

                    // Get the column we're interested in
                    let column = batch.column(field_index);
                    let row_offset = total_rows - batch_rows;

                    // Process the column for all rows
                    let new_strings =
                        process_column_for_all_rows(column, row_offset, &mut value_to_doc_ids)?;
                    strings_seen += new_strings;

                    // Log progress periodically
                    if batch_count % 5 == 0 {
                        println!(
                            "Processed {} batches, {} rows, {} unique values so far",
                            batch_count,
                            total_rows,
                            value_to_doc_ids.len()
                        );
                    }
                }
                Ok(None) => {
                    println!("Batch {} was None, skipping", batch_idx);
                }
                Err(e) => {
                    eprintln!("Error reading batch {}: {}", batch_idx, e);
                }
            }
        }

        println!(
            "Processed {} batches with {} total rows",
            batch_count, total_rows
        );
        println!("Found {} total string values", strings_seen);
        println!("Resulting in {} unique values", value_to_doc_ids.len());

        let process_time = process_start.elapsed();

        // Update stats
        stats.add_timing(setup_time, process_time);
        stats = stats.finish(value_to_doc_ids.len());
        stats.print_benchmark();

        Ok((value_to_doc_ids, stats))
    } else {
        Err(ArrowError::InvalidArgumentError(
            "Failed to read the first batch to get schema".to_string(),
        ))
    }
}

/// Retrieves field values by specific document IDs using zero-copy IPC
fn get_field_values_by_doc_ids_zero_copy(
    field_name: &str,
    doc_ids: &[usize],
    file_path: &str,
) -> Result<(HashMap<String, Vec<usize>>, QueryStats), ArrowError> {
    // Create stats object
    let mut stats = QueryStats::new(
        "get_field_values_by_doc_ids_zero_copy",
        field_name,
        doc_ids.len(),
    );

    let mut value_to_doc_ids: HashMap<String, Vec<usize>> = HashMap::new();

    // Measure setup time
    let setup_start = Instant::now();

    // Open the file and memory map it
    let file = File::open(file_path)
        .map_err(|e| ArrowError::InvalidArgumentError(format!("Failed to open file: {}", e)))?;

    // Memory mapping allows us to access file contents without loading entire file into memory
    let mmap = unsafe {
        Mmap::map(&file).map_err(|e| {
            ArrowError::InvalidArgumentError(format!("Failed to memory map file: {}", e))
        })?
    };

    // Use the zero-copy approach
    // We need to copy from the mmap to handle lifetime issues
    let bytes = Bytes::copy_from_slice(&mmap[..]);
    let buffer = Buffer::from(bytes);

    // Create the IPCBufferDecoder which will efficiently decode record batches
    let decoder = IPCBufferDecoder::new(buffer);
    println!("Number of batches in file: {}", decoder.num_batches());

    // Sort and deduplicate document IDs for efficient processing
    let mut sorted_doc_ids = doc_ids.to_vec();
    sorted_doc_ids.sort_unstable();
    sorted_doc_ids.dedup();

    // Track which document IDs we've processed
    let mut processed_ids = vec![false; sorted_doc_ids.len()];
    let mut processed_doc_ids = 0;
    let mut row_offset = 0;
    let mut batch_count = 0;

    let setup_time = setup_start.elapsed();

    // Processing phase - measure time
    let process_start = Instant::now();

    // We need to get the first batch to access its schema
    if let Ok(Some(first_batch)) = decoder.get_batch(0) {
        let schema = first_batch.schema();
        let field_index = schema.index_of(field_name).map_err(|_| {
            ArrowError::InvalidArgumentError(format!("Field '{}' not found in schema", field_name))
        })?;

        // Process batches in sequence
        for batch_idx in 0..decoder.num_batches() {
            match decoder.get_batch(batch_idx) {
                Ok(Some(batch)) => {
                    batch_count += 1;
                    let batch_rows = batch.num_rows();

                    // Find which document IDs are in this batch
                    let mut batch_doc_ids = Vec::new();
                    let batch_indices = Vec::new();

                    for (i, &doc_id) in sorted_doc_ids.iter().enumerate() {
                        let effective_doc_id = doc_id.saturating_sub(row_offset);
                        if !processed_ids[i]
                            && doc_id >= row_offset
                            && effective_doc_id < batch_rows
                        {
                            batch_doc_ids.push((i, effective_doc_id, doc_id));
                            processed_ids[i] = true;
                            processed_doc_ids += 1;
                        }
                    }

                    // If we found any document IDs in this batch, process them
                    if !batch_doc_ids.is_empty() {
                        let column = batch.column(field_index).clone();

                        // Process based on the data type
                        if let DataType::List(_) = column.data_type() {
                            let mut filter = vec![false; batch_rows];
                            for &doc_id in &batch_doc_ids {
                                filter[doc_id.1] = true;
                            }

                            process_list_array_batch_with_filter(
                                &column,
                                &filter,
                                &batch_doc_ids
                                    .iter()
                                    .map(|&(_, _, doc_id)| doc_id)
                                    .collect::<Vec<_>>(),
                                &batch_indices,
                                &sorted_doc_ids,
                                &mut value_to_doc_ids,
                            )?;
                        } else {
                            let mut filter = vec![false; batch_rows];
                            for &doc_id in &batch_doc_ids {
                                filter[doc_id.1] = true;
                            }

                            process_scalar_array_batch_with_filter(
                                &column,
                                &filter,
                                &batch_doc_ids
                                    .iter()
                                    .map(|&(_, _, doc_id)| doc_id)
                                    .collect::<Vec<_>>(),
                                &batch_indices,
                                &sorted_doc_ids,
                                &mut value_to_doc_ids,
                            )?;
                        }
                    }

                    row_offset += batch_rows;

                    // If we've processed all document IDs, we can stop
                    if processed_doc_ids == sorted_doc_ids.len() {
                        println!("All document IDs processed after {} batches", batch_count);
                        break;
                    }
                }
                Ok(None) => {
                    println!("Batch {} was None, skipping", batch_idx);
                }
                Err(e) => {
                    eprintln!("Error reading batch {}: {}", batch_idx, e);
                }
            }
        }

        println!(
            "Processed {} batches with {} total rows",
            batch_count, row_offset
        );
        println!(
            "Processed {} out of {} document IDs",
            processed_doc_ids,
            sorted_doc_ids.len()
        );

        let process_time = process_start.elapsed();

        // Update stats
        stats.add_timing(setup_time, process_time);
        stats = stats.finish(value_to_doc_ids.len());
        stats.print_benchmark();

        Ok((value_to_doc_ids, stats))
    } else {
        Err(ArrowError::InvalidArgumentError(
            "Failed to read the first batch to get schema".to_string(),
        ))
    }
}

/// Print a sample batch to verify schema
fn print_sample_batch() {
    println!("\n=== Sample Batch Content ===");

    // Create a small batch with just a few records
    match create_large_record_batch(5) {
        Ok(batch) => {
            let schema = batch.schema();

            println!("Schema: (total {} fields)", schema.fields().len());
            for (i, field) in schema.fields().iter().enumerate() {
                println!("  {}. {} ({})", i, field.name(), field.data_type());
            }

            println!("\nData rows: (total {} rows)", batch.num_rows());
            for row_idx in 0..batch.num_rows() {
                println!("Row {}:", row_idx);

                for col_idx in 0..batch.num_columns() {
                    let field = schema.field(col_idx);
                    let column = batch.column(col_idx);

                    print!("  {}: ", field.name());

                    if !column.is_null(row_idx) {
                        match field.data_type() {
                            DataType::Int64 => {
                                let array = column.as_primitive::<Int64Type>();
                                println!("{}", array.value(row_idx));
                            }
                            DataType::Int32 => {
                                let array = column.as_primitive::<Int32Type>();
                                println!("{}", array.value(row_idx));
                            }
                            DataType::Utf8 => {
                                let array = column.as_any().downcast_ref::<StringArray>().unwrap();
                                println!("{}", array.value(row_idx));
                            }
                            DataType::Boolean => {
                                let array = column.as_boolean();
                                println!("{}", array.value(row_idx));
                            }
                            DataType::Timestamp(_, _) => {
                                println!("Timestamp value (in milliseconds)");
                            }
                            DataType::List(field_ref) => {
                                if let Some(list_array) =
                                    column.as_any().downcast_ref::<GenericListArray<i32>>()
                                {
                                    if !list_array.is_null(row_idx) {
                                        let values = list_array.value(row_idx);
                                        let values_len = values.len();

                                        print!("[");
                                        match field_ref.data_type() {
                                            DataType::Utf8 => {
                                                if let Some(string_array) =
                                                    values.as_any().downcast_ref::<StringArray>()
                                                {
                                                    for i in 0..values_len {
                                                        if i > 0 {
                                                            print!(", ");
                                                        }
                                                        print!("\"{}\"", string_array.value(i));
                                                    }
                                                }
                                            }
                                            DataType::Boolean => {
                                                if let Some(bool_array) =
                                                    values.as_any().downcast_ref::<BooleanArray>()
                                                {
                                                    for i in 0..values_len {
                                                        if i > 0 {
                                                            print!(", ");
                                                        }
                                                        print!("{}", bool_array.value(i));
                                                    }
                                                }
                                            }
                                            _ => print!("... {} items", values_len),
                                        }
                                        println!("]");
                                    } else {
                                        println!("[]");
                                    }
                                } else {
                                    println!("[list values]");
                                }
                            }
                            _ => println!("[unsupported type]"),
                        }
                    } else {
                        println!("null");
                    }
                }
                println!();
            }
        }
        Err(e) => eprintln!("Error creating sample batch: {}", e),
    }

    println!("==============================\n");
}

fn main() {
    let pid = process::id();
    println!("\n🚀 Process started with PID: {}", pid);
    println!("To monitor in real-time, run in another terminal:");
    println!("  top -pid {}", pid);
    println!("  # or");
    println!("  ps -o pid,rss,vsz,command -p {}\n", pid);

    let num_rows: usize = 15_000_000;

    // Print initial memory usage
    print_memory_stats("Program Start");

    // Collect benchmark stats
    let mut all_benchmark_stats: Vec<QueryStats> = Vec::new();

    // Ask if user wants to see a sample batch
    println!("\nDo you want to see a sample batch? (y/n)");
    let mut show_sample = String::new();
    std::io::stdin().read_line(&mut show_sample).unwrap();
    if show_sample.trim().to_lowercase() == "y" {
        print_sample_batch();
    }

    // Create a large Arrow file if it doesn't exist
    let arrow_file_path = "large_segment.arrow";
    if !std::path::Path::new(arrow_file_path).exists() {
        println!("Creating large Arrow file...");
        let start = Instant::now();

        // Create a large record batch with 1 million rows
        match create_large_record_batch(num_rows) {
            Ok(batch) => {
                // Calculate in-memory size of the Arrow data
                let arrow_memory_size = batch.get_array_memory_size();
                println!(
                    "Arrow in-memory size: {} bytes ({:.2} MB)",
                    arrow_memory_size,
                    arrow_memory_size as f64 / (1024.0 * 1024.0)
                );

                // Write the batch to an Arrow IPC file
                match write_batch_ipc(arrow_file_path, &batch) {
                    Ok(_) => {
                        let creation_time = start.elapsed();
                        println!(
                            "Created Arrow file with {} rows in {:?}",
                            num_rows, creation_time
                        );

                        // Get and print the Arrow file size
                        if let Ok(metadata) = std::fs::metadata(arrow_file_path) {
                            let arrow_file_size = metadata.len();
                            println!("\n=== Arrow File Size Statistics ===");
                            println!(
                                "Arrow IPC file size: {} bytes ({:.2} MB)",
                                arrow_file_size,
                                arrow_file_size as f64 / (1024.0 * 1024.0)
                            );

                            // Compare with in-memory size
                            println!(
                                "Arrow in-memory size: {} bytes ({:.2} MB)",
                                arrow_memory_size,
                                arrow_memory_size as f64 / (1024.0 * 1024.0)
                            );

                            // Calculate storage efficiency
                            let storage_ratio = arrow_file_size as f64 / arrow_memory_size as f64;
                            println!(
                                "Storage ratio (file size / in-memory size): {:.2}",
                                storage_ratio
                            );

                            // Calculate bytes per row
                            println!(
                                "Average bytes per row in Arrow file: {:.2} bytes",
                                arrow_file_size as f64 / num_rows as f64
                            );
                            println!("==============================\n");
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to write Arrow file: {}", e);
                        return;
                    }
                }
            }
            Err(e) => {
                eprintln!("Failed to create record batch: {}", e);
                return;
            }
        }

        print_memory_stats("After creating Arrow file");
    } else {
        println!("Arrow file already exists, skipping creation");

        // Print the size of the existing Arrow file
        if let Ok(metadata) = std::fs::metadata(arrow_file_path) {
            let arrow_file_size = metadata.len();
            println!("\n=== Arrow File Size Statistics ===");
            println!(
                "Arrow IPC file size: {} bytes ({:.2} MB)",
                arrow_file_size,
                arrow_file_size as f64 / (1024.0 * 1024.0)
            );
            println!(
                "Average bytes per row in Arrow file: {:.2} bytes",
                arrow_file_size as f64 / num_rows as f64
            );
            println!("==============================\n");
        }
    }

    // Ask user which field to process
    println!("\nSelect a field to process:");
    println!("1. level (log level)");
    println!("2. source.region (region information)");
    println!("3. user.id (user identifier)");
    println!("4. tags (list of tags)");
    println!("5. user.metrics.active (boolean flag)");

    let mut field_choice = String::new();
    std::io::stdin().read_line(&mut field_choice).unwrap();
    let field_choice = field_choice.trim();

    let field_name = match field_choice {
        "1" => "level",
        "2" => "source.region",
        "3" => "user.id",
        "4" => "tags",
        "5" => "user.metrics.active",
        _ => "level", // default
    };

    println!("Processing field: {}", field_name);

    // Ask the user which test to run
    println!("\nSelect a test to run (1-7) or 'all' to run all tests:");
    println!("1. Process all documents with zero-copy IPC (most memory efficient)");
    println!("2. Process specific document IDs with zero-copy IPC");
    println!("3. Calculate numeric statistics with zero-copy IPC");

    let mut choice = String::new();
    std::io::stdin().read_line(&mut choice).unwrap();
    let choice = choice.trim();

    let tests_to_run = if choice.to_lowercase() == "all" {
        (1..=3).collect::<Vec<_>>()
    } else if let Ok(test_num) = choice.parse::<i32>() {
        if (1..=3).contains(&test_num) {
            vec![test_num]
        } else {
            vec![1] // Default to zero-copy test
        }
    } else {
        vec![1] // Default to zero-copy test
    };

    for test in &tests_to_run {
        match test {
            1 => {
                // Process all documents using zero-copy IPC
                println!("\n=== Zero-Copy IPC Processing (All Documents) ===");
                print_memory_stats("Before processing with zero-copy");

                let start = Instant::now();
                match get_field_values_zero_copy(field_name, arrow_file_path) {
                    Ok((results, stats)) => {
                        let elapsed = start.elapsed();
                        println!("All documents processed in {:?}", elapsed);
                        println!("Number of unique values: {}", results.len());

                        // Print total document count
                        let total_docs: usize = results.values().map(|ids| ids.len()).sum();
                        println!("Total document references: {}", total_docs);

                        // Print first few results
                        println!("\nFirst few values:");
                        for (value, ids) in results.iter().take(5) {
                            println!(
                                "  {} -> {} documents (first few: {:?})",
                                value,
                                ids.len(),
                                &ids[..std::cmp::min(5, ids.len())]
                            );
                        }

                        // Calculate rows per second
                        let rows_per_second = num_rows as f64 / elapsed.as_secs_f64();
                        println!("\nPerformance: {:.2} rows/second", rows_per_second);

                        // Add benchmark stats
                        all_benchmark_stats.push(stats);
                    }
                    Err(e) => eprintln!("Error processing with zero-copy: {}", e),
                }

                print_memory_stats("After processing with zero-copy");
            }
            2 => {
                // Generate random document IDs
                let mut rng = thread_rng();
                let total_docs = num_rows; // Total number of documents to select from
                let num_docs = 100; // Number of documents to process (increased for better measurement)
                let mut doc_ids: Vec<usize> = (0..total_docs).collect();
                doc_ids.shuffle(&mut rng);
                let doc_ids = doc_ids[..num_docs].to_vec();

                println!(
                    "Selected {} random documents from a pool of {}",
                    num_docs, total_docs
                );

                // Process the field using the zero-copy approach for specific doc_ids
                println!("\n=== Zero-Copy IPC Processing (Specific Document IDs) ===");
                print_memory_stats("Before processing with zero-copy");

                let start = Instant::now();
                match get_field_values_by_doc_ids_zero_copy(field_name, &doc_ids, arrow_file_path) {
                    Ok((results, stats)) => {
                        let elapsed = start.elapsed();
                        println!("Field processed in {:?}", elapsed);
                        println!("Number of unique values: {}", results.len());

                        // Print first few results
                        println!("\nFirst few values:");
                        for (value, ids) in results.iter().take(5) {
                            println!(
                                "  {} -> {} documents (first few: {:?})",
                                value,
                                ids.len(),
                                &ids[..std::cmp::min(5, ids.len())]
                            );
                        }

                        // Calculate ops per second (documents processed)
                        let docs_per_second = num_docs as f64 / elapsed.as_secs_f64();
                        println!("\nPerformance: {:.2} documents/second", docs_per_second);

                        // Add benchmark stats
                        all_benchmark_stats.push(stats);
                    }
                    Err(e) => eprintln!("Error processing with zero-copy: {}", e),
                }

                print_memory_stats("After processing with zero-copy");
            }
            3 => {
                // This test is only applicable for numeric fields
                let numeric_field_name = if field_name != "user.metrics.login_time_ms"
                    && field_name != "user.metrics.clicks"
                    && field_name != "payload_size"
                {
                    println!("\nTest 7 requires a numeric field. Please choose user.metrics.login_time_ms, user.metrics.clicks, or payload_size.");
                    println!("Using payload_size for this test.");
                    "payload_size"
                } else {
                    field_name
                };

                // Generate random document IDs
                let mut rng = thread_rng();
                let total_docs = num_rows; // Total number of documents to select from
                let num_docs = 1000; // Number of documents to process (increased for better measurement)
                let mut doc_ids: Vec<usize> = (0..total_docs).collect();
                doc_ids.shuffle(&mut rng);
                let doc_ids = doc_ids[..num_docs].to_vec();

                println!(
                    "Selected {} random documents from a pool of {} for numeric stats",
                    num_docs, total_docs
                );

                // Calculate numeric stats using zero-copy approach
                println!("\n=== Zero-Copy IPC Numeric Stats Processing ===");
                print_memory_stats("Before processing");

                let start = Instant::now();
                match get_numeric_stats_by_doc_ids_zero_copy::<i64>(
                    numeric_field_name,
                    &doc_ids,
                    arrow_file_path,
                ) {
                    Ok(stats_result) => {
                        let elapsed = start.elapsed();
                        println!("Numeric stats calculated in {:?}", elapsed);
                        println!(
                            "\nStatistics for field '{}' across {} documents:",
                            numeric_field_name, num_docs
                        );
                        println!("  Sum: {}", stats_result.sum);
                        println!("  Min: {}", stats_result.min);
                        println!("  Max: {}", stats_result.max);
                        println!(
                            "  Average: {:.2}",
                            stats_result.sum as f64 / num_docs as f64
                        );

                        // Calculate ops per second
                        let docs_per_second = num_docs as f64 / elapsed.as_secs_f64();
                        println!("\nPerformance: {:.2} documents/second", docs_per_second);

                        // Create and add benchmark stats
                        let mut query_stats = QueryStats::new(
                            "get_numeric_stats_by_doc_ids_zero_copy",
                            numeric_field_name,
                            num_docs,
                        );
                        query_stats.add_timing(Duration::from_secs(0), elapsed);
                        query_stats = query_stats.finish(3); // 3 stats: sum, min, max
                        all_benchmark_stats.push(query_stats);
                    }
                    Err(e) => eprintln!("Error calculating numeric stats: {}", e),
                }

                print_memory_stats("After processing");
            }
            _ => {
                // Test options 2, 3, and 6 are not implemented in this shorter version
                println!("Test {} not implemented in this shorter version", test);
            }
        }
    }

    // Print benchmark results
    if !all_benchmark_stats.is_empty() {
        println!("\n=== BENCHMARK RESULTS ===");
        print_benchmark_table(&all_benchmark_stats);
    }

    // Keep the program alive for final monitoring
    println!("\nPress Enter to exit...");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
}

/// Statistics for numeric fields, supporting both scalar and list fields
/// For list fields, statistics are calculated across all values in the lists
#[derive(Debug)]
pub struct NumericStats<T> {
    pub sum: T,
    pub max: T,
    pub min: T,
}

/// Supports both scalar numeric fields and lists of numeric values
/// Generic type T must implement numeric traits for calculations
fn get_numeric_stats_by_doc_ids_zero_copy<T>(
    field_name: &str,
    doc_ids: &[usize],
    file_path: &str,
) -> Result<NumericStats<T>, ArrowError>
where
    T: 'static
        + std::fmt::Debug
        + num::traits::Num
        + num::traits::FromPrimitive
        + std::cmp::Ord
        + std::cmp::PartialOrd
        + Copy,
{
    // Initialize with default values
    let mut sum = T::zero();
    let mut max = None;
    let mut min = None;

    // Open the file and memory map it
    let file = File::open(file_path)
        .map_err(|e| ArrowError::InvalidArgumentError(format!("Failed to open file: {}", e)))?;

    // Memory mapping allows us to access file contents without loading entire file into memory
    let mmap = unsafe {
        Mmap::map(&file).map_err(|e| {
            ArrowError::InvalidArgumentError(format!("Failed to memory map file: {}", e))
        })?
    };

    // Use the zero-copy approach
    // We need to copy from the mmap to handle lifetime issues
    let bytes = Bytes::copy_from_slice(&mmap[..]);
    // Create Buffer from Bytes (zero-copy)
    let buffer = Buffer::from(bytes);

    // Create the IPCBufferDecoder which will efficiently decode record batches
    let decoder = IPCBufferDecoder::new(buffer);
    println!("Number of batches in file: {}", decoder.num_batches());

    // Sort and deduplicate document IDs for efficient processing
    let mut sorted_doc_ids = doc_ids.to_vec();
    sorted_doc_ids.sort_unstable();
    sorted_doc_ids.dedup();

    // Track which document IDs we've processed
    let mut processed_ids = vec![false; sorted_doc_ids.len()];
    let mut processed_doc_ids = 0;
    let mut row_offset = 0;
    let mut batch_count = 0;

    // We need to get the first batch to access its schema
    if let Ok(Some(first_batch)) = decoder.get_batch(0) {
        let schema = first_batch.schema();
        let field_index = schema.index_of(field_name).map_err(|_| {
            ArrowError::InvalidArgumentError(format!("Field '{}' not found in schema", field_name))
        })?;

        // Check if this is a numeric field
        let field = schema.field(field_index);
        let data_type = field.data_type();

        match data_type {
            DataType::Int32 | DataType::Int64 | DataType::List(_) => {
                // Valid numeric type, continue
            }
            _ => {
                return Err(ArrowError::InvalidArgumentError(format!(
                    "Field '{}' is not a numeric type or list of numeric values",
                    field_name
                )));
            }
        }

        // Process batches in sequence
        for batch_idx in 0..decoder.num_batches() {
            match decoder.get_batch(batch_idx) {
                Ok(Some(batch)) => {
                    batch_count += 1;
                    let batch_rows = batch.num_rows();

                    // Find which document IDs are in this batch
                    let mut batch_doc_ids = Vec::new();

                    for (i, &doc_id) in sorted_doc_ids.iter().enumerate() {
                        let effective_doc_id = doc_id.saturating_sub(row_offset);
                        if !processed_ids[i]
                            && doc_id >= row_offset
                            && effective_doc_id < batch_rows
                        {
                            batch_doc_ids.push((i, effective_doc_id, doc_id));
                            processed_ids[i] = true;
                            processed_doc_ids += 1;
                        }
                    }

                    // If we found any document IDs in this batch, process them
                    if !batch_doc_ids.is_empty() {
                        let column = batch.column(field_index);

                        // Calculate statistics based on the data type
                        match column.data_type() {
                            DataType::Int32 => {
                                let array = column.as_primitive::<Int32Type>();

                                for (_i, effective_doc_id, _original_doc_id) in &batch_doc_ids {
                                    if !array.is_null(*effective_doc_id) {
                                        let value = T::from_i32(array.value(*effective_doc_id))
                                            .ok_or_else(|| {
                                                ArrowError::InvalidArgumentError(
                                                    "Failed to convert Int32 to target type"
                                                        .to_string(),
                                                )
                                            })?;

                                        sum = sum + value;
                                        max = Some(max.map_or(value, |m| std::cmp::max(m, value)));
                                        min = Some(min.map_or(value, |m| std::cmp::min(m, value)));
                                    }
                                }
                            }
                            DataType::Int64 => {
                                let array = column.as_primitive::<Int64Type>();

                                for (_i, effective_doc_id, _original_doc_id) in &batch_doc_ids {
                                    if !array.is_null(*effective_doc_id) {
                                        let value = T::from_i64(array.value(*effective_doc_id))
                                            .ok_or_else(|| {
                                                ArrowError::InvalidArgumentError(
                                                    "Failed to convert Int64 to target type"
                                                        .to_string(),
                                                )
                                            })?;

                                        sum = sum + value;
                                        max = Some(max.map_or(value, |m| std::cmp::max(m, value)));
                                        min = Some(min.map_or(value, |m| std::cmp::min(m, value)));
                                    }
                                }
                            }
                            DataType::List(field) => match field.data_type() {
                                DataType::Int32 => {
                                    let list_array: &GenericListArray<i32> =
                                        column.as_any().downcast_ref().ok_or_else(|| {
                                            ArrowError::InvalidArgumentError(
                                                "Failed to downcast to list array".to_string(),
                                            )
                                        })?;

                                    for (_i, effective_doc_id, _original_doc_id) in &batch_doc_ids {
                                        if !list_array.is_null(*effective_doc_id) {
                                            let values = list_array.value(*effective_doc_id);
                                            let int_array = values.as_primitive::<Int32Type>();

                                            for i in 0..int_array.len() {
                                                if !int_array.is_null(i) {
                                                    let value = T::from_i32(int_array.value(i)).ok_or_else(|| {
                                                        ArrowError::InvalidArgumentError(
                                                            "Failed to convert Int32 to target type".to_string(),
                                                        )
                                                    })?;

                                                    sum = sum + value;
                                                    max = Some(max.map_or(value, |m| {
                                                        std::cmp::max(m, value)
                                                    }));
                                                    min = Some(min.map_or(value, |m| {
                                                        std::cmp::min(m, value)
                                                    }));
                                                }
                                            }
                                        }
                                    }
                                }
                                DataType::Int64 => {
                                    let list_array: &GenericListArray<i32> =
                                        column.as_any().downcast_ref().ok_or_else(|| {
                                            ArrowError::InvalidArgumentError(
                                                "Failed to downcast to list array".to_string(),
                                            )
                                        })?;

                                    for (_i, effective_doc_id, _original_doc_id) in &batch_doc_ids {
                                        if !list_array.is_null(*effective_doc_id) {
                                            let values = list_array.value(*effective_doc_id);
                                            let int_array = values.as_primitive::<Int64Type>();

                                            for i in 0..int_array.len() {
                                                if !int_array.is_null(i) {
                                                    let value = T::from_i64(int_array.value(i)).ok_or_else(|| {
                                                        ArrowError::InvalidArgumentError(
                                                            "Failed to convert Int64 to target type".to_string(),
                                                        )
                                                    })?;

                                                    sum = sum + value;
                                                    max = Some(max.map_or(value, |m| {
                                                        std::cmp::max(m, value)
                                                    }));
                                                    min = Some(min.map_or(value, |m| {
                                                        std::cmp::min(m, value)
                                                    }));
                                                }
                                            }
                                        }
                                    }
                                }
                                _ => {
                                    return Err(ArrowError::InvalidArgumentError(
                                        "List field must contain numeric values".to_string(),
                                    ));
                                }
                            },
                            _ => {
                                return Err(ArrowError::InvalidArgumentError(
                                    "Field must be a numeric type or a list of numeric values"
                                        .to_string(),
                                ));
                            }
                        }
                    }

                    row_offset += batch_rows;

                    // If we've processed all document IDs, we can stop
                    if processed_doc_ids == sorted_doc_ids.len() {
                        println!("All document IDs processed after {} batches", batch_count);
                        break;
                    }
                }
                Ok(None) => {
                    println!("Batch {} was None, skipping", batch_idx);
                }
                Err(e) => {
                    eprintln!("Error reading batch {}: {}", batch_idx, e);
                }
            }
        }

        println!(
            "Processed {} batches, {} out of {} document IDs",
            batch_count,
            processed_doc_ids,
            sorted_doc_ids.len()
        );
    } else {
        return Err(ArrowError::InvalidArgumentError(
            "Failed to read the first batch to get schema".to_string(),
        ));
    }

    Ok(NumericStats {
        sum,
        max: max.unwrap_or_else(T::zero),
        min: min.unwrap_or_else(T::zero),
    })
}

/// Process a column for all rows
/// Returns the number of string values processed
fn process_column_for_all_rows(
    array: &ArrayRef,
    row_offset: usize,
    value_to_doc_ids: &mut HashMap<String, Vec<usize>>,
) -> Result<usize, ArrowError> {
    let mut strings_processed = 0;

    match array.data_type() {
        DataType::List(field) => {
            match field.data_type() {
                DataType::Utf8 => {
                    let list_array: &GenericListArray<i32> =
                        array.as_any().downcast_ref().ok_or_else(|| {
                            ArrowError::InvalidArgumentError(
                                "Failed to downcast to list array".to_string(),
                            )
                        })?;

                    // Process all rows in the list array
                    for doc_id in 0..list_array.len() {
                        let global_doc_id = row_offset + doc_id;

                        if !list_array.is_null(doc_id) {
                            let values = list_array.value(doc_id);
                            let string_array = values
                                .as_any()
                                .downcast_ref::<StringArray>()
                                .ok_or_else(|| {
                                    ArrowError::InvalidArgumentError(
                                        "Failed to downcast to string array".to_string(),
                                    )
                                })?;

                            for j in 0..string_array.len() {
                                if !string_array.is_null(j) {
                                    let value = string_array.value(j).to_string();
                                    strings_processed += 1;

                                    // Use entry API to avoid duplicate lookups
                                    value_to_doc_ids
                                        .entry(value)
                                        .or_insert_with(|| Vec::with_capacity(8))
                                        .push(global_doc_id);
                                }
                            }
                        }
                    }
                }
                DataType::Int32 => {
                    let list_array: &GenericListArray<i32> =
                        array.as_any().downcast_ref().ok_or_else(|| {
                            ArrowError::InvalidArgumentError(
                                "Failed to downcast to list array".to_string(),
                            )
                        })?;

                    // Process all rows in the list array
                    for doc_id in 0..list_array.len() {
                        let global_doc_id = row_offset + doc_id;

                        if !list_array.is_null(doc_id) {
                            let values = list_array.value(doc_id);
                            let int_array = values.as_primitive::<Int32Type>();

                            for j in 0..int_array.len() {
                                if !int_array.is_null(j) {
                                    let value = int_array.value(j).to_string();
                                    strings_processed += 1;

                                    value_to_doc_ids
                                        .entry(value)
                                        .or_insert_with(|| Vec::with_capacity(8))
                                        .push(global_doc_id);
                                }
                            }
                        }
                    }
                }
                _ => {
                    // For other types, use the generic approach
                    if let Some(list_array) = array.as_any().downcast_ref::<GenericListArray<i32>>()
                    {
                        for doc_id in 0..list_array.len() {
                            let global_doc_id = row_offset + doc_id;

                            if !list_array.is_null(doc_id) {
                                let values = list_array.value(doc_id);
                                let len = values.len();

                                for j in 0..len {
                                    if !values.is_null(j) {
                                        let value = format!("{:?}", values.as_any());
                                        strings_processed += 1;

                                        value_to_doc_ids
                                            .entry(value)
                                            .or_insert_with(|| Vec::with_capacity(8))
                                            .push(global_doc_id);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        _ => {
            // For scalar arrays (non-list)
            let num_rows = array.len();

            for doc_id in 0..num_rows {
                let global_doc_id = row_offset + doc_id;

                if !array.is_null(doc_id) {
                    let value = match array.data_type() {
                        DataType::Utf8 => {
                            let string_array =
                                array.as_any().downcast_ref::<StringArray>().unwrap();
                            string_array.value(doc_id).to_string()
                        }
                        DataType::Boolean => {
                            let bool_array = array.as_boolean();
                            bool_array.value(doc_id).to_string()
                        }
                        DataType::Int32 => {
                            let int_array = array.as_primitive::<Int32Type>();
                            int_array.value(doc_id).to_string()
                        }
                        DataType::Int64 => {
                            let int_array = array.as_primitive::<Int64Type>();
                            int_array.value(doc_id).to_string()
                        }
                        _ => format!("{:?}", array.as_any()),
                    };

                    strings_processed += 1;

                    value_to_doc_ids
                        .entry(value)
                        .or_insert_with(|| Vec::with_capacity(8))
                        .push(global_doc_id);
                }
            }
        }
    }

    Ok(strings_processed)
}
