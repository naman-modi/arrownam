use arrow::array::{
    Array, ArrayRef, AsArray, GenericListArray, Int32Array, Int64Array, StringArray,
};
use arrow::datatypes::{DataType, Int32Type, Int64Type};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use std::collections::HashMap;

/// Statistics for numeric fields
#[derive(Debug)]
pub struct NumericStats<T> {
    pub sum: T,
    pub max: T,
    pub min: T,
}

/// Gets field values by document IDs
pub fn get_field_values_by_doc_ids(
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

/// Gets numeric statistics by document IDs
pub fn get_numeric_stats_by_doc_ids<T: 'static + std::fmt::Debug>(
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

    match data_type {
        DataType::Int32 => process_int32_stats(column, doc_ids),
        DataType::Int64 => process_int64_stats(column, doc_ids),
        DataType::List(field) => match field.data_type() {
            DataType::Int32 => process_int32_list_stats(column, doc_ids),
            DataType::Int64 => process_int64_list_stats(column, doc_ids),
            _ => Err(ArrowError::InvalidArgumentError(
                "List field must contain numeric values".to_string(),
            )),
        },
        _ => Err(ArrowError::InvalidArgumentError(
            "Field must be a numeric type or a list of numeric values".to_string(),
        )),
    }
}

fn process_list_array(
    array: &ArrayRef,
    doc_ids: &[usize],
    value_to_doc_ids: &mut HashMap<String, Vec<usize>>,
) -> Result<(), ArrowError> {
    match array.data_type() {
        DataType::List(field) => match field.data_type() {
            DataType::Utf8 => process_string_list(array, doc_ids, value_to_doc_ids),
            DataType::Int32 => process_int32_list(array, doc_ids, value_to_doc_ids),
            _ => process_generic_list(array, doc_ids, value_to_doc_ids),
        },
        _ => Err(ArrowError::InvalidArgumentError(
            "Expected a list array".to_string(),
        )),
    }
}

fn process_string_list(
    array: &ArrayRef,
    doc_ids: &[usize],
    value_to_doc_ids: &mut HashMap<String, Vec<usize>>,
) -> Result<(), ArrowError> {
    let list_array: &GenericListArray<i32> = array.as_any().downcast_ref().ok_or_else(|| {
        ArrowError::InvalidArgumentError("Failed to downcast to list array".to_string())
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
                ArrowError::InvalidArgumentError("Failed to downcast to string array".to_string())
            })?;

        for i in 0..string_array.len() {
            if !string_array.is_null(i) {
                let value = string_array.value(i).to_string();
                value_to_doc_ids.entry(value).or_default().push(doc_id);
            }
        }
    }

    Ok(())
}

fn process_int32_list(
    array: &ArrayRef,
    doc_ids: &[usize],
    value_to_doc_ids: &mut HashMap<String, Vec<usize>>,
) -> Result<(), ArrowError> {
    let list_array: &GenericListArray<i32> = array.as_any().downcast_ref().ok_or_else(|| {
        ArrowError::InvalidArgumentError("Failed to downcast to list array".to_string())
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

    Ok(())
}

fn process_generic_list(
    array: &ArrayRef,
    doc_ids: &[usize],
    value_to_doc_ids: &mut HashMap<String, Vec<usize>>,
) -> Result<(), ArrowError> {
    if let Some(list_array) = array.as_any().downcast_ref::<GenericListArray<i32>>() {
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
        Ok(())
    } else {
        Err(ArrowError::InvalidArgumentError(
            "Failed to process generic list array".to_string(),
        ))
    }
}

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

fn process_int32_stats<T>(
    array: &ArrayRef,
    doc_ids: &[usize],
) -> Result<NumericStats<T>, ArrowError>
where
    T: num::traits::Num + num::traits::FromPrimitive + std::cmp::Ord + std::cmp::PartialOrd + Copy,
{
    let array = array.as_any().downcast_ref::<Int32Array>().ok_or_else(|| {
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

fn process_int64_stats<T>(
    array: &ArrayRef,
    doc_ids: &[usize],
) -> Result<NumericStats<T>, ArrowError>
where
    T: num::traits::Num + num::traits::FromPrimitive + std::cmp::Ord + std::cmp::PartialOrd + Copy,
{
    let array = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
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

fn process_int32_list_stats<T>(
    array: &ArrayRef,
    doc_ids: &[usize],
) -> Result<NumericStats<T>, ArrowError>
where
    T: num::traits::Num + num::traits::FromPrimitive + std::cmp::Ord + std::cmp::PartialOrd + Copy,
{
    let list_array: &GenericListArray<i32> = array.as_any().downcast_ref().ok_or_else(|| {
        ArrowError::InvalidArgumentError("Failed to downcast to list array".to_string())
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

fn process_int64_list_stats<T>(
    array: &ArrayRef,
    doc_ids: &[usize],
) -> Result<NumericStats<T>, ArrowError>
where
    T: num::traits::Num + num::traits::FromPrimitive + std::cmp::Ord + std::cmp::PartialOrd + Copy,
{
    let list_array: &GenericListArray<i32> = array.as_any().downcast_ref().ok_or_else(|| {
        ArrowError::InvalidArgumentError("Failed to downcast to list array".to_string())
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
