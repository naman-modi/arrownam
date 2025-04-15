mod dns;
mod ipc;
mod stats;

use arrow::array::Int32Array;

fn main() {
    let _ = arrow_array();
    let _ = array_array_builder();

    match dns::builder::create_dns_record_batch() {
        Ok(temp_batch) => {
            println!("Created record batch with {} rows", temp_batch.num_rows());
            println!("Number of cols: {}", temp_batch.num_columns());

            ipc::write_batch_ipc("segment.arrow", &temp_batch).unwrap();

            let read_batches = ipc::read_batches_ipc("segment.arrow").unwrap();
            let batch = &read_batches[0];

            let doc_ids = vec![0, 2];

            // Works with list of strings (domain names)
            match stats::get_field_values_by_doc_ids(
                "responseData.answers.domainName",
                batch,
                &doc_ids,
            ) {
                Ok(domain_results) => println!("Domain names: {:?}", domain_results),
                Err(e) => eprintln!("Error getting domain names: {}", e),
            }

            // Works with boolean values
            match stats::get_field_values_by_doc_ids("isError", batch, &doc_ids) {
                Ok(error_results) => println!("Error flags: {:?}", error_results),
                Err(e) => eprintln!("Error getting error flags: {}", e),
            }

            // Works with list of integers (time responses)
            match stats::get_field_values_by_doc_ids(
                "responseData.answers.timeResponse",
                batch,
                &doc_ids,
            ) {
                Ok(time_results) => println!("Time responses: {:?}", time_results),
                Err(e) => eprintln!("Error getting time responses: {}", e),
            }

            // Calculate numeric stats for time responses
            match stats::get_numeric_stats_by_doc_ids::<i32>(
                "responseData.answers.timeResponse",
                batch,
                &doc_ids,
            ) {
                Ok(stats) => println!("Time response stats: {:?}", stats),
                Err(e) => eprintln!("Error calculating time response stats: {}", e),
            }

            // Calculate numeric stats for message type ID
            match stats::get_numeric_stats_by_doc_ids::<i64>("messageTypeId", batch, &doc_ids) {
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
