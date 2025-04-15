use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, TimeUnit};

/// DNS record schema definition
pub fn create_dns_schema() -> Schema {
    let domain_name_array_field = Field::new(
        "responseData.answers.domainName",
        DataType::List(Arc::new(Field::new("item", DataType::Utf8, true))),
        true,
    );

    let rcode_name_field = Field::new("responseData.rcodeName", DataType::Utf8, false);

    let time_response_field = Field::new(
        "responseData.answers.timeResponse",
        DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
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

/// DNS record batch builder
pub mod builder {
    use super::*;
    use arrow::array::{
        ArrayRef, BooleanArray, Int32Builder, Int64Builder, ListBuilder, StringBuilder,
        TimestampMicrosecondArray,
    };
    use arrow::record_batch::RecordBatch;

    /// Creates a sample DNS record batch for testing
    pub fn create_dns_record_batch() -> arrow::error::Result<RecordBatch> {
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
}
