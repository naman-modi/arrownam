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
use std::collections::HashMap;
use std::fs::File;
use std::io::Cursor;
use std::sync::Arc;

fn write_batch_ipc(path: &str, batch: &RecordBatch) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = FileWriter::try_new(file, &batch.schema()).unwrap();
    writer.write(batch).unwrap();
    writer.finish().unwrap();
    Ok(())
}

/// Memory-mapped file reader for Arrow IPC format
/// Uses memory mapping for efficient file reading without loading entire file into memory
fn read_batches_ipc(path: &str) -> ArrowResult<Vec<RecordBatch>> {
    let file = File::open(path)?;
    // Memory mapping allows us to access file contents without loading it entirely into memory
    let mmap_new = unsafe { Mmap::map(&file)? };
    // Cursor provides Seek trait implementation for the byte slice
    let cursor = Cursor::new(&mmap_new[..]);
    let reader = FileReader::try_new(Box::new(cursor), None)?;
    reader.collect::<ArrowResult<Vec<_>>>()
}

fn main() {
    let _ = arrow_array();
    let _ = array_array_builder();

    match create_dns_record_batch() {
        Ok(temp_batch) => {
            println!("Created record batch with {} rows", temp_batch.num_rows());
            println!("Number of cols: {}", temp_batch.num_columns());

            write_batch_ipc("segment.arrow", &temp_batch).unwrap();

            let read_batches = read_batches_ipc("segment.arrow").unwrap();
            let batch = &read_batches[0];

            let doc_ids = vec![0, 2];

            // Works with list of strings (domain names)
            match get_field_values_by_doc_ids("responseData.answers.domainName", &batch, &doc_ids) {
                Ok(domain_results) => println!("Domain names: {:?}", domain_results),
                Err(e) => eprintln!("Error getting domain names: {}", e),
            }

            // Works with boolean values
            match get_field_values_by_doc_ids("isError", &batch, &doc_ids) {
                Ok(error_results) => println!("Error flags: {:?}", error_results),
                Err(e) => eprintln!("Error getting error flags: {}", e),
            }

            // Works with list of integers (time responses)
            match get_field_values_by_doc_ids("responseData.answers.timeResponse", &batch, &doc_ids)
            {
                Ok(time_results) => println!("Time responses: {:?}", time_results),
                Err(e) => eprintln!("Error getting time responses: {}", e),
            }

            // Calculate numeric stats for time responses
            match get_numeric_stats_by_doc_ids::<i32>(
                "responseData.answers.timeResponse",
                &batch,
                &doc_ids,
            ) {
                Ok(stats) => println!("Time response stats: {:?}", stats),
                Err(e) => eprintln!("Error calculating time response stats: {}", e),
            }

            // Calculate numeric stats for message type ID
            match get_numeric_stats_by_doc_ids::<i64>("messageTypeId", &batch, &doc_ids) {
                Ok(stats) => println!("Message type ID stats: {:?}", stats),
                Err(e) => eprintln!("Error calculating message type ID stats: {}", e),
            }
        }
        Err(e) => eprintln!("Error creating record batch: {}", e),
    }
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

/// Creates a sample DNS record batch with 3 records demonstrating different scenarios:
/// 1. Successful query with multiple answers
/// 2. Successful query with single answer
/// 3. Failed query with error
fn create_dns_record_batch() -> ArrowResult<RecordBatch> {
    // Create the schema
    let schema = create_dns_schema();

    // Create builders for our arrays
    let mut domain_names_builder = ListBuilder::new(StringBuilder::new());
    let mut rcode_name_builder = StringBuilder::new();
    let mut time_response_builder = ListBuilder::new(Int32Builder::new());
    let mut message_type_builder = Int64Builder::new();
    let mut timestamp_builder = TimestampMicrosecondArray::builder(0);
    let mut is_error_builder = BooleanArray::builder(0);

    // Add data for 3 records
    // Record 1
    domain_names_builder.values().append_value("example.com");
    domain_names_builder
        .values()
        .append_value("sub.example.com");
    domain_names_builder.append(true);

    rcode_name_builder.append_value("NOERROR");

    time_response_builder.values().append_value(100);
    time_response_builder.values().append_value(150);
    time_response_builder.append(true);

    message_type_builder.append_value(1);
    timestamp_builder.append_value(1234567890000000);
    is_error_builder.append_value(false);

    // Record 2
    domain_names_builder.values().append_value("test.com");
    domain_names_builder.append(true);

    rcode_name_builder.append_value("NOERROR");

    time_response_builder.values().append_value(200);
    time_response_builder.append(true);

    message_type_builder.append_value(1);
    timestamp_builder.append_value(1234567891000000);
    is_error_builder.append_value(false);

    // Record 3
    domain_names_builder.values().append_value("error.com");
    domain_names_builder.append(true);

    rcode_name_builder.append_value("SERVFAIL");

    time_response_builder.values().append_value(500);
    time_response_builder.append(true);

    message_type_builder.append_value(2);
    timestamp_builder.append_value(1234567892000000);
    is_error_builder.append_value(true);

    // Create arrays from builders
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(domain_names_builder.finish()),
        Arc::new(rcode_name_builder.finish()),
        Arc::new(time_response_builder.finish()),
        Arc::new(message_type_builder.finish()),
        Arc::new(timestamp_builder.finish()),
        Arc::new(is_error_builder.finish()),
    ];

    // Create and return the record batch
    RecordBatch::try_new(Arc::new(schema), arrays)
}

/// Retrieves field values by document IDs, supporting both scalar and list fields
/// Returns a mapping of values to the document IDs that contain them
///
/// For list fields (e.g., domain names), each value in the list is mapped to its document ID
/// For scalar fields (e.g., isError), the single value is mapped to its document ID
fn get_field_values_by_doc_ids(
    field_name: &str,
    batch: &RecordBatch,
    doc_ids: &[usize],
) -> Result<HashMap<String, Vec<usize>>, ArrowError> {
    let mut value_to_doc_ids: HashMap<String, Vec<usize>> = HashMap::new();

    let column_index = batch.schema().index_of(field_name).map_err(|_| {
        ArrowError::InvalidArgumentError(format!("Field '{}' not found in schema", field_name))
    })?;

    let column = batch.column(column_index);

    // Check if this is a list type
    if let DataType::List(_) = column.data_type() {
        process_list_array(column, doc_ids, &mut value_to_doc_ids)?;
    } else {
        process_scalar_array(column, doc_ids, &mut value_to_doc_ids)?;
    }

    Ok(value_to_doc_ids)
}

/// Processes list arrays (e.g., lists of domain names or time responses)
/// Handles different types of list elements (strings, integers, etc.)
/// Maps each value in the list to its containing document ID
fn process_list_array(
    array: &ArrayRef,
    doc_ids: &[usize],
    value_to_doc_ids: &mut HashMap<String, Vec<usize>>,
) -> Result<(), ArrowError> {
    // Use the appropriate list array type based on the data type
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

                    for &doc_id in doc_ids {
                        if doc_id >= list_array.len() {
                            continue;
                        }
                        let values = list_array.value(doc_id);
                        let string_array = values
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .ok_or_else(|| {
                                ArrowError::InvalidArgumentError(
                                    "Failed to downcast to string array".to_string(),
                                )
                            })?;

                        for i in 0..string_array.len() {
                            if !string_array.is_null(i) {
                                let value = string_array.value(i).to_string();
                                value_to_doc_ids.entry(value).or_default().push(doc_id);
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

                    for &doc_id in doc_ids {
                        if doc_id >= list_array.len() {
                            continue;
                        }

                        let values = list_array.value(doc_id);
                        let int_array = values.as_primitive::<Int32Type>();

                        for i in 0..int_array.len() {
                            if !int_array.is_null(i) {
                                let value = int_array.value(i).to_string();
                                value_to_doc_ids.entry(value).or_default().push(doc_id);
                            }
                        }
                    }
                }
                _ => {
                    // For other types, use the generic approach
                    if let Some(list_array) = array.as_any().downcast_ref::<GenericListArray<i32>>()
                    {
                        for &doc_id in doc_ids {
                            if doc_id >= list_array.len() {
                                continue;
                            }

                            let values = list_array.value(doc_id);
                            let len = values.len();

                            for i in 0..len {
                                if !values.is_null(i) {
                                    let value = format!("{:?}", values.as_any());
                                    value_to_doc_ids.entry(value).or_default().push(doc_id);
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

/// Processes scalar arrays (e.g., boolean flags or single values)
/// Maps each value to its document ID
fn process_scalar_array(
    array: &ArrayRef,
    indices: &[usize],
    value_to_doc_ids: &mut HashMap<String, Vec<usize>>,
) -> Result<(), ArrowError> {
    for &idx in indices {
        if idx < array.len() && !array.is_null(idx) {
            let value = match array.data_type() {
                DataType::Utf8 => {
                    let string_array = array.as_any().downcast_ref::<StringArray>().unwrap();
                    string_array.value(idx).to_string()
                }
                DataType::Boolean => {
                    let bool_array = array.as_boolean();
                    bool_array.value(idx).to_string()
                }
                DataType::Int32 => {
                    let int_array = array.as_primitive::<Int32Type>();
                    int_array.value(idx).to_string()
                }
                DataType::Int64 => {
                    let int_array = array.as_primitive::<Int64Type>();
                    int_array.value(idx).to_string()
                }
                _ => format!("{:?}", array.as_any()),
            };

            value_to_doc_ids.entry(value).or_default().push(idx);
        }
    }

    Ok(())
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
