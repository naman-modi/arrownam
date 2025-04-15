use arrow::array::{
    Array, ArrayRef, AsArray, BooleanArray, GenericListArray, Int32Array, Int32Builder, Int64Array,
    Int64Builder, ListBuilder, StringArray, StringBuilder, TimestampMicrosecondArray,
};
use arrow::datatypes::{DataType, Field, Int32Type, Int64Type, Schema, TimeUnit};
use arrow::error::{ArrowError, Result as ArrowResult};
use arrow::ipc::reader::FileReader;
use arrow::ipc::writer::FileWriter;
use arrow::record_batch::RecordBatch;
use memmap2::Mmap;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::HashMap;
use std::fs::File;
use std::io::Cursor;
use std::process;
use std::sync::Arc;
use std::thread::sleep;
use std::time::{Duration, Instant};

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

/// Creates a schema for DNS records with nested fields
/// The schema supports:
/// - Lists of domain names (nested under responseData.answers)
/// - Response codes
/// - Lists of time responses (nested under responseData.answers)
/// - Message type IDs
/// - Timestamps
/// - Error flags
fn create_dns_schema() -> Schema {
    let domain_name_array_field = Field::new(
        "responseData.answers.domainName",
        DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
        true,
    );

    let rcode_name_field = Field::new("responseData.rcodeName", DataType::Utf8, false);

    let time_response_field = Field::new(
        "responseData.answers.timeResponse",
        DataType::List(Arc::new(Field::new("item", DataType::Int32, true))), // Changed to true
        true,
    );

    let message_type_id = Field::new("messageTypeId", DataType::Int64, false);

    let timestamp = Field::new(
        "timestamp",
        DataType::Timestamp(TimeUnit::Microsecond, None),
        false,
    );

    let is_error = Field::new("isError", DataType::Boolean, false);

    Schema::new(vec![
        domain_name_array_field,
        rcode_name_field,
        time_response_field,
        message_type_id,
        timestamp,
        is_error,
    ])
}

/// Creates a large record batch with specified number of rows
fn create_large_record_batch(num_rows: usize) -> ArrowResult<RecordBatch> {
    let schema = create_dns_schema();

    // Create builders with larger capacity
    let mut domain_names_builder = ListBuilder::new(StringBuilder::new());
    let mut rcode_name_builder = StringBuilder::new();
    let mut time_response_builder = ListBuilder::new(Int32Builder::new());
    let mut message_type_builder = Int64Builder::new();
    let mut timestamp_builder = TimestampMicrosecondArray::builder(num_rows);
    let mut is_error_builder = BooleanArray::builder(num_rows);

    // For tracking raw data size
    let mut raw_string_size = 0;
    let mut raw_int_size = 0;
    let mut domain_count = 0;

    // Generate large amount of data
    for i in 0..num_rows {
        // Add multiple domain names per record to increase size
        let domain1 = format!("domain{}.com", i % 2);
        let domain2 = format!("sub{}.domain{}.com", i % 3, i % 4);
        let domain3 = format!("sub2{}.domain{}.com", i % 5, i % 6);

        // Track raw string size (actual bytes used for the strings)
        raw_string_size += domain1.len();
        raw_string_size += domain2.len();
        raw_string_size += domain3.len();
        domain_count += 3;

        domain_names_builder.values().append_value(domain1);
        domain_names_builder.values().append_value(domain2);
        domain_names_builder.values().append_value(domain3);
        domain_names_builder.append(true);

        // Track raw string size for rcode_name
        raw_string_size += "NOERROR".len();
        rcode_name_builder.append_value("NOERROR");

        // Add multiple time responses
        time_response_builder
            .values()
            .append_value(100 + (i as i32));
        time_response_builder
            .values()
            .append_value(150 + (i as i32));
        time_response_builder
            .values()
            .append_value(200 + (i as i32));
        time_response_builder.append(true);

        // Track raw int size (12 bytes for 3 integers)
        raw_int_size += 3 * std::mem::size_of::<i32>();

        message_type_builder.append_value(i as i64);
        raw_int_size += std::mem::size_of::<i64>();

        timestamp_builder.append_value(1234567890000000 + (i as i64));
        raw_int_size += std::mem::size_of::<i64>();

        is_error_builder.append_value(i % 5 == 0); // Some errors
        raw_int_size += 1; // 1 byte for boolean
    }

    // Calculate and print raw data size statistics
    let total_raw_size = raw_string_size + raw_int_size;
    println!("\n=== Raw Data Size Statistics ===");
    println!("Number of rows: {}", num_rows);
    println!("Number of domain strings: {}", domain_count);
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
        Arc::new(domain_names_builder.finish()),
        Arc::new(rcode_name_builder.finish()),
        Arc::new(time_response_builder.finish()),
        Arc::new(message_type_builder.finish()),
        Arc::new(timestamp_builder.finish()),
        Arc::new(is_error_builder.finish()),
    ];

    RecordBatch::try_new(Arc::new(schema), arrays)
}

/// Retrieves field values by document IDs from an Arrow IPC file using a memory-efficient approach
/// This implementation is more memory efficient as it processes only the needed documents
fn get_field_values_by_doc_ids(
    field_name: &str,
    doc_ids: &[usize],
    file_path: &str,
) -> Result<HashMap<String, Vec<usize>>, ArrowError> {
    let mut value_to_doc_ids: HashMap<String, Vec<usize>> = HashMap::new();

    sleep(Duration::from_secs(5));

    // Open the file and memory map it
    let file = File::open(file_path)
        .map_err(|e| ArrowError::InvalidArgumentError(format!("Failed to open file: {}", e)))?;

    // Get file metadata to calculate chunk sizes and offsets
    let file_size = file
        .metadata()
        .map_err(|e| {
            ArrowError::InvalidArgumentError(format!("Failed to get file metadata: {}", e))
        })?
        .len();

    println!("File size: {} bytes", file_size);

    // Sort and deduplicate document IDs for efficient processing
    let mut sorted_doc_ids = doc_ids.to_vec();
    sorted_doc_ids.sort_unstable();
    sorted_doc_ids.dedup();

    // Track which document IDs we've processed
    let mut processed_ids = vec![false; sorted_doc_ids.len()];
    let mut processed_doc_ids = 0;

    // Process each batch
    let mut batch_count = 0;
    let mut total_rows = 0;

    // Memory mapping allows us to access file contents without loading entire file into memory
    let mmap = unsafe {
        Mmap::map(&file).map_err(|e| {
            ArrowError::InvalidArgumentError(format!("Failed to memory map file: {}", e))
        })?
    };

    // Keep the mmap alive for the duration of the function
    let _buf = &mmap[..];
    let cursor = Cursor::new(_buf);
    let reader = FileReader::try_new(Box::new(cursor), None)?;

    // Get the schema to find the field index
    let schema = reader.schema();
    let field_index = schema.index_of(field_name).map_err(|_| {
        ArrowError::InvalidArgumentError(format!("Field '{}' not found in schema", field_name))
    })?;

    for batch_result in reader {
        match batch_result {
            Ok(batch) => {
                batch_count += 1;
                let batch_rows = batch.num_rows();
                total_rows += batch_rows;

                // Create a filter bitset to identify which rows in this batch we need
                let mut filter = vec![false; batch_rows];
                let mut has_docs_in_batch = false;

                // Find which document IDs are in this batch
                let mut batch_doc_ids = Vec::new();
                let mut batch_indices = Vec::new();

                for (i, &doc_id) in sorted_doc_ids.iter().enumerate() {
                    if !processed_ids[i] && doc_id < batch_rows {
                        filter[doc_id] = true;
                        batch_doc_ids.push(doc_id);
                        batch_indices.push(i);
                        processed_ids[i] = true;
                        processed_doc_ids += 1;
                        has_docs_in_batch = true;
                    }
                }

                // If we found any document IDs in this batch, process them
                if has_docs_in_batch {
                    // Extract only the column we need and clone it
                    let column = batch.columns()[field_index].clone();
                    // Drop the full batch to free memory
                    drop(batch);

                    // Process based on the data type
                    if let DataType::List(_) = column.data_type() {
                        process_list_array_batch_with_filter(
                            &column,
                            &filter,
                            &batch_doc_ids,
                            &batch_indices,
                            &sorted_doc_ids,
                            &mut value_to_doc_ids,
                        )?;
                    } else {
                        process_scalar_array_batch_with_filter(
                            &column,
                            &filter,
                            &batch_doc_ids,
                            &batch_indices,
                            &sorted_doc_ids,
                            &mut value_to_doc_ids,
                        )?;
                    }
                }

                // If we've processed all document IDs, we can stop
                if processed_doc_ids == sorted_doc_ids.len() {
                    println!("All document IDs processed after {} batches", batch_count);
                    break;
                }
            }
            Err(e) => eprintln!("Error reading batch {}: {}", batch_count, e),
        }
    }

    sleep(Duration::from_secs(5));

    println!(
        "Processed {} batches with {} total rows",
        batch_count, total_rows
    );
    println!(
        "Processed {} out of {} document IDs",
        processed_doc_ids,
        sorted_doc_ids.len()
    );

    Ok(value_to_doc_ids)
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

/// Retrieves field values for all documents in an Arrow IPC file using a memory-efficient approach
/// This implementation processes all documents in the file with minimal memory usage
fn get_all_field_values(
    field_name: &str,
    file_path: &str,
) -> Result<HashMap<String, Vec<usize>>, ArrowError> {
    // Reserve a large initial capacity for the HashMap to avoid frequent resizing

    let mut value_to_doc_ids: HashMap<String, Vec<usize>> = HashMap::with_capacity(1_000_000);

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

    // Keep the mmap alive for the duration of the function
    let _buf = &mmap[..];
    let cursor = Cursor::new(_buf);
    let reader = FileReader::try_new(Box::new(cursor), None)?;

    // Get the schema to find the field index
    let schema = reader.schema();
    let field_index = schema.index_of(field_name).map_err(|_| {
        ArrowError::InvalidArgumentError(format!("Field '{}' not found in schema", field_name))
    })?;

    // Process batches
    let mut batch_count = 0;
    let mut total_rows = 0;
    let mut processed_rows = 0;
    let mut strings_seen = 0;
    let chunk_size = 1_000_000; // Process in chunks of 1 million rows

    // Process batches one at a time to minimize memory usage
    for batch_result in reader {
        match batch_result {
            Ok(batch) => {
                batch_count += 1;
                let batch_rows = batch.num_rows();
                total_rows += batch_rows;

                // Extract only the column we need and clone it
                let column = &batch.columns()[field_index];

                // Process the column for all rows
                let new_strings =
                    process_column_for_all_rows(column, processed_rows, &mut value_to_doc_ids)?;
                strings_seen += new_strings;

                // Update the processed row count
                processed_rows += batch_rows;

                drop(batch)
            }
            Err(e) => eprintln!("Error reading batch {}: {}", batch_count, e),
        }
    }

    println!(
        "Processed {} batches with {} total rows",
        batch_count, total_rows
    );
    println!("Found {} total string values", strings_seen);
    println!("Resulting in {} unique values", value_to_doc_ids.len());

    Ok(value_to_doc_ids)
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

/// Retrieves field values for a range of rows in an Arrow IPC file using a memory-efficient approach
/// This allows scanning only a subset of documents which is useful for very large datasets
fn get_field_values_by_row_range(
    field_name: &str,
    file_path: &str,
    start_row: usize,
    end_row: Option<usize>,
) -> Result<HashMap<String, Vec<usize>>, ArrowError> {
    // Reserve a reasonable initial capacity for the HashMap
    let mut value_to_doc_ids: HashMap<String, Vec<usize>> = HashMap::with_capacity(100_000);

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

    // Keep the mmap alive for the duration of the function
    let _buf = &mmap[..];
    let cursor = Cursor::new(_buf);
    let reader = FileReader::try_new(Box::new(cursor), None)?;

    // Get the schema to find the field index
    let schema = reader.schema();
    let field_index = schema.index_of(field_name).map_err(|_| {
        ArrowError::InvalidArgumentError(format!("Field '{}' not found in schema", field_name))
    })?;

    // Process batches
    let mut batch_count = 0;
    let mut total_rows = 0;
    let mut processed_rows = 0;
    let mut strings_seen = 0;
    let mut current_row = 0;

    // Process batches one at a time to minimize memory usage
    for batch_result in reader {
        match batch_result {
            Ok(batch) => {
                batch_count += 1;
                let batch_rows = batch.num_rows();
                total_rows += batch_rows;

                // Skip batches before start_row
                if current_row + batch_rows <= start_row {
                    current_row += batch_rows;
                    continue;
                }

                // Stop if we've gone past end_row
                if let Some(end) = end_row {
                    if current_row >= end {
                        break;
                    }
                }

                // Calculate the effective batch start and end
                let batch_start = start_row.saturating_sub(current_row);

                let batch_end = if let Some(end) = end_row {
                    if current_row + batch_rows > end {
                        end - current_row
                    } else {
                        batch_rows
                    }
                } else {
                    batch_rows
                };

                if batch_start >= batch_end {
                    current_row += batch_rows;
                    continue;
                }

                // Extract only the column we need and clone it
                let column = &batch.columns()[field_index];
                // Drop the full batch to free memory

                // Process the range within the column
                let new_strings = process_column_row_range(
                    column,
                    current_row + batch_start,
                    batch_start,
                    batch_end,
                    &mut value_to_doc_ids,
                )?;

                strings_seen += new_strings;
                processed_rows += batch_end - batch_start;

                drop(batch);

                // // Log progress periodically
                // if batch_count % 10 == 0 {
                //     println!(
                //         "Processed {} batches, {} rows, {} unique values so far",
                //         batch_count,
                //         processed_rows,
                //         value_to_doc_ids.len()
                //     );
                //     print_memory_stats(&format!("After processing {} batches", batch_count));
                // }

                current_row += batch_rows;

                // Stop if we've reached the end_row
                if let Some(end) = end_row {
                    if current_row >= end {
                        break;
                    }
                }
            }
            Err(e) => eprintln!("Error reading batch {}: {}", batch_count, e),
        }
    }

    println!(
        "Processed {} batches with {} total rows in range [{}, {})",
        batch_count,
        processed_rows,
        start_row,
        end_row.unwrap_or(total_rows)
    );
    println!("Found {} total string values", strings_seen);
    println!("Resulting in {} unique values", value_to_doc_ids.len());

    Ok(value_to_doc_ids)
}

/// Process a column for a range of rows
/// Returns the number of string values processed
fn process_column_row_range(
    array: &ArrayRef,
    global_row_offset: usize,
    start_row: usize,
    end_row: usize,
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

                    // Process rows in the specified range
                    for row_idx in start_row..end_row {
                        let global_doc_id = global_row_offset + (row_idx - start_row);

                        if !list_array.is_null(row_idx) {
                            let values = list_array.value(row_idx);
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

                    // Process rows in the specified range
                    for row_idx in start_row..end_row {
                        let global_doc_id = global_row_offset + (row_idx - start_row);

                        if !list_array.is_null(row_idx) {
                            let values = list_array.value(row_idx);
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
                        for row_idx in start_row..end_row {
                            let global_doc_id = global_row_offset + (row_idx - start_row);

                            if !list_array.is_null(row_idx) {
                                let values = list_array.value(row_idx);
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

            for row_idx in start_row..std::cmp::min(end_row, num_rows) {
                let global_doc_id = global_row_offset + (row_idx - start_row);

                if !array.is_null(row_idx) {
                    let value = match array.data_type() {
                        DataType::Utf8 => {
                            let string_array =
                                array.as_any().downcast_ref::<StringArray>().unwrap();
                            string_array.value(row_idx).to_string()
                        }
                        DataType::Boolean => {
                            let bool_array = array.as_boolean();
                            bool_array.value(row_idx).to_string()
                        }
                        DataType::Int32 => {
                            let int_array = array.as_primitive::<Int32Type>();
                            int_array.value(row_idx).to_string()
                        }
                        DataType::Int64 => {
                            let int_array = array.as_primitive::<Int64Type>();
                            int_array.value(row_idx).to_string()
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

    // Field to process
    let field_name = "responseData.answers.domainName";
    println!("Processing field: {}", field_name);

    // Ask the user which test to run
    println!("\nSelect a test to run:");
    println!("1. Process specific document IDs (efficient for sparse queries)");
    println!("2. Process all documents (efficient for full scans)");
    println!("3. Process a range of rows (efficient for partitioned processing)");

    let mut choice = String::new();
    std::io::stdin().read_line(&mut choice).unwrap();
    let choice = choice.trim();

    match choice {
        "1" => {
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

            // Process the field using the memory-optimized approach for specific doc_ids
            println!("\n=== Memory-Optimized Processing (Specific Document IDs) ===");
            print_memory_stats("Before processing");

            let start = Instant::now();
            match get_field_values_by_doc_ids(field_name, &doc_ids, arrow_file_path) {
                Ok(results) => {
                    let elapsed = start.elapsed();
                    println!("Field processed in {:?}", elapsed);
                    println!("Number of unique values: {}", results.len());

                    // Print first few results
                    println!("\nFirst few values:");
                    for (value, ids) in results.iter().take(5) {
                        println!("  {} -> documents: {:?}", value, ids);
                    }

                    // Calculate ops per second (documents processed)
                    let docs_per_second = num_docs as f64 / elapsed.as_secs_f64();
                    println!("\nPerformance: {:.2} documents/second", docs_per_second);
                }
                Err(e) => eprintln!("Error processing field: {}", e),
            }

            print_memory_stats("After processing");
        }
        "2" => {
            // Process all documents
            println!("\n=== Memory-Optimized Processing (All Documents) ===");
            print_memory_stats("Before processing all documents");

            sleep(Duration::from_secs(5));

            let start = Instant::now();
            match get_all_field_values(field_name, arrow_file_path) {
                Ok(results) => {
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

                    drop(results);
                }
                Err(e) => eprintln!("Error processing all documents: {}", e),
            }

            sleep(Duration::from_secs(5));

            print_memory_stats("After processing all documents");
        }
        "3" => {
            // Process a range of rows
            println!("\nEnter start row (0-based index):");
            let mut start_row_input = String::new();
            std::io::stdin().read_line(&mut start_row_input).unwrap();
            let start_row: usize = start_row_input.trim().parse().unwrap_or(0);

            println!("Enter end row (leave empty for all rows from start):");
            let mut end_row_input = String::new();
            std::io::stdin().read_line(&mut end_row_input).unwrap();
            let end_row: Option<usize> = if end_row_input.trim().is_empty() {
                None
            } else {
                Some(end_row_input.trim().parse().unwrap_or(num_rows))
            };

            println!(
                "\n=== Memory-Optimized Processing (Row Range: {} to {}) ===",
                start_row,
                end_row.map_or("end".to_string(), |v| v.to_string())
            );
            print_memory_stats("Before processing row range");

            let start = Instant::now();
            match get_field_values_by_row_range(field_name, arrow_file_path, start_row, end_row) {
                Ok(results) => {
                    let elapsed = start.elapsed();
                    println!("Row range processed in {:?}", elapsed);
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
                    let processed_rows = end_row.unwrap_or(num_rows).saturating_sub(start_row);
                    let rows_per_second = processed_rows as f64 / elapsed.as_secs_f64();
                    println!("\nPerformance: {:.2} rows/second", rows_per_second);
                }
                Err(e) => eprintln!("Error processing row range: {}", e),
            }

            print_memory_stats("After processing row range");
        }
        _ => {
            println!("Invalid choice, exiting.");
            return;
        }
    }

    // Keep the program alive for final monitoring
    println!("\nPress Enter to exit...");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
}

fn arrow_array() -> Int32Array {
    let array = Int32Array::from(vec![Some(2), None, Some(5), None]);
    assert_eq!(array.value(0), 2);
    array
}

fn array_array_builder() -> Int32Array {
    let mut array_builder = Int32Array::builder(100);
    array_builder.append_value(1);
    array_builder.append_null();
    array_builder.append_value(3);
    array_builder.append_value(4);
    array_builder.finish()
}

/// Statistics for numeric fields, supporting both scalar and list fields
/// For list fields, statistics are calculated across all values in the lists
#[derive(Debug)]
pub struct NumericStats<T> {
    pub sum: T,
    pub max: T,
    pub min: T,
}

/// Calculates numeric statistics for a field across specified document IDs
/// Supports both scalar numeric fields and lists of numeric values
/// Generic type T must implement numeric traits for calculations
fn get_numeric_stats_by_doc_ids<T: 'static + std::fmt::Debug>(
    field_name: &str,
    batch: &RecordBatch,
    doc_ids: &[usize],
) -> Result<NumericStats<T>, ArrowError>
where
    T: num::traits::Num + num::traits::FromPrimitive + std::cmp::Ord + std::cmp::PartialOrd + Copy,
{
    let column_index = batch.schema().index_of(field_name).map_err(|_| {
        ArrowError::InvalidArgumentError(format!("Field '{}' not found in schema", field_name))
    })?;

    let column = batch.column(column_index);
    let data_type = column.data_type();

    // Check if this is a numeric type
    match data_type {
        DataType::Int32 => {
            let array = column
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| {
                    ArrowError::InvalidArgumentError("Failed to downcast to Int32Array".to_string())
                })?;

            let mut sum = T::zero();
            let mut max = None;
            let mut min = None;

            for &doc_id in doc_ids {
                if doc_id < array.len() && !array.is_null(doc_id) {
                    let value = T::from_i32(array.value(doc_id)).ok_or_else(|| {
                        ArrowError::InvalidArgumentError(
                            "Failed to convert Int32 to target type".to_string(),
                        )
                    })?;

                    sum = sum + value;
                    max = Some(max.map_or(value, |m| std::cmp::max(m, value)));
                    min = Some(min.map_or(value, |m| std::cmp::min(m, value)));
                }
            }

            Ok(NumericStats {
                sum,
                max: max.unwrap_or_else(T::zero),
                min: min.unwrap_or_else(T::zero),
            })
        }
        DataType::Int64 => {
            let array = column
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| {
                    ArrowError::InvalidArgumentError("Failed to downcast to Int64Array".to_string())
                })?;

            let mut sum = T::zero();
            let mut max = None;
            let mut min = None;

            for &doc_id in doc_ids {
                if doc_id < array.len() && !array.is_null(doc_id) {
                    let value = T::from_i64(array.value(doc_id)).ok_or_else(|| {
                        ArrowError::InvalidArgumentError(
                            "Failed to convert Int64 to target type".to_string(),
                        )
                    })?;

                    sum = sum + value;
                    max = Some(max.map_or(value, |m| std::cmp::max(m, value)));
                    min = Some(min.map_or(value, |m| std::cmp::min(m, value)));
                }
            }

            Ok(NumericStats {
                sum,
                max: max.unwrap_or_else(T::zero),
                min: min.unwrap_or_else(T::zero),
            })
        }
        DataType::List(field) => match field.data_type() {
            DataType::Int32 => {
                let list_array: &GenericListArray<i32> =
                    column.as_any().downcast_ref().ok_or_else(|| {
                        ArrowError::InvalidArgumentError(
                            "Failed to downcast to list array".to_string(),
                        )
                    })?;

                let mut sum = T::zero();
                let mut max = None;
                let mut min = None;

                for &doc_id in doc_ids {
                    if doc_id < list_array.len() {
                        let values = list_array.value(doc_id);
                        let int_array = values.as_primitive::<Int32Type>();

                        for i in 0..int_array.len() {
                            if !int_array.is_null(i) {
                                let value = T::from_i32(int_array.value(i)).ok_or_else(|| {
                                    ArrowError::InvalidArgumentError(
                                        "Failed to convert Int32 to target type".to_string(),
                                    )
                                })?;

                                sum = sum + value;
                                max = Some(max.map_or(value, |m| std::cmp::max(m, value)));
                                min = Some(min.map_or(value, |m| std::cmp::min(m, value)));
                            }
                        }
                    }
                }

                Ok(NumericStats {
                    sum,
                    max: max.unwrap_or_else(T::zero),
                    min: min.unwrap_or_else(T::zero),
                })
            }
            DataType::Int64 => {
                let list_array: &GenericListArray<i32> =
                    column.as_any().downcast_ref().ok_or_else(|| {
                        ArrowError::InvalidArgumentError(
                            "Failed to downcast to list array".to_string(),
                        )
                    })?;

                let mut sum = T::zero();
                let mut max = None;
                let mut min = None;

                for &doc_id in doc_ids {
                    if doc_id < list_array.len() {
                        let values = list_array.value(doc_id);
                        let int_array = values.as_primitive::<Int64Type>();

                        for i in 0..int_array.len() {
                            if !int_array.is_null(i) {
                                let value = T::from_i64(int_array.value(i)).ok_or_else(|| {
                                    ArrowError::InvalidArgumentError(
                                        "Failed to convert Int64 to target type".to_string(),
                                    )
                                })?;

                                sum = sum + value;
                                max = Some(max.map_or(value, |m| std::cmp::max(m, value)));
                                min = Some(min.map_or(value, |m| std::cmp::min(m, value)));
                            }
                        }
                    }
                }

                Ok(NumericStats {
                    sum,
                    max: max.unwrap_or_else(T::zero),
                    min: min.unwrap_or_else(T::zero),
                })
            }
            _ => Err(ArrowError::InvalidArgumentError(
                "List field must contain numeric values".to_string(),
            )),
        },
        _ => Err(ArrowError::InvalidArgumentError(
            "Field must be a numeric type or a list of numeric values".to_string(),
        )),
    }
}
