# Arrow Zero-Copy Benchmark

A high-performance log data processing system built with Rust and Apache Arrow for efficient zero-copy querying of large log datasets.

## Performance Metrics

### Data Processing

* **Large dataset** of log data in Arrow IPC format
* **Zero-copy memory access** for minimized memory footprint
* **Efficient columnar processing** for faster query execution

### Query Performance

The system demonstrates excellent performance across various query types:


| Query Type                     | Field                | Doc IDs  | Result Rows | Total Time | Peak Memory Impact (in MB)  |
|--------------------------------|----------------------|----------|-------------|------------|---------------|
| get_field_values_zero_copy     | tags                 | 10M        | 10          | 3.98    s | 542.06       |
| get_field_values_by_doc_ids_zero_copy | level                | 100      | 4           | 1.03    s |  84.00         |
| get_field_values_zero_copy     | source.region        | 10M        | 3           | 0.61    s |  84.95         |
| get_numeric_stats_zero_copy    | payload_size         | 10M        | 3           | 0.80    s | 135.75          |
| get_numeric_stats_for_100_random_docs | payload_size         | 100      | 3           | 0.29    s | 80.00         |
| get_field_values_zero_copy     | tags (2nd time)               | 10M        | 10          | 1.24    s | 138.27        |



## Query API

The system implements a high-level query API with zero-copy memory efficiency:

### Field Value Queries

1. **Get Field Values Using Zero-Copy**  
   `get_field_values_zero_copy(field_name, file_path)`  
   Retrieves all values for a specified field across the dataset using memory mapping for zero-copy access.

2. **Get Field Values by Document IDs Using Zero-Copy**  
   `get_field_values_by_doc_ids_zero_copy(field_name, doc_ids, file_path)`  
   Retrieves specific field values for given document IDs with zero-copy memory mapping.

3. **Get Field Values by Row Range Using Zero-Copy**  
   `get_field_values_by_row_range_zero_copy(field_name, file_path, start_row, end_row)`  
   Retrieves field values within a specific row range using zero-copy access.

### Numeric Statistic Queries

1. **Get Numeric Stats by Document IDs Using Zero-Copy**  
   `get_numeric_stats_by_doc_ids_zero_copy(field_name, doc_ids, file_path)`  
   Calculates statistics (min, max, sum) for a numeric field across specified document IDs with zero-copy memory mapping.

## Architecture

The system consists of the following components:

### Data Model

* **Schema**: Flexible schema for log data with support for nested and complex types
* **Arrow Arrays**: Efficient columnar storage with native support for complex types
* **Zero-copy IPC**: Memory-mapped Arrow IPC files for minimal memory usage

### Key Optimizations

1. **Zero-Copy Memory Mapping**  
   * Direct memory access without copying data  
   * Lower memory footprint for large datasets
   
2. **Efficient Array Processing**  
   * Specialized functions for different data types  
   * Optimized handling of list arrays and nested data
   
3. **Query Optimizations**  
   * Document ID deduplication and sorting  
   * Batch-by-batch processing for memory efficiency  
   * Early termination when target document IDs are fulfilled

## Usage Example

```rust
// Generate sample data
let num_rows = 15_000_000;
let arrow_file_path = "large_segment.arrow";

// Example: Get all values for a field
match get_field_values_zero_copy("level", arrow_file_path) {
    Ok((results, stats)) => {
        println!("Found {} unique values", results.len());
        stats.print_benchmark();
    }
    Err(e) => eprintln!("Error: {}", e),
}

// Example: Get values for specific document IDs
let doc_ids: Vec<usize> = (0..100).map(|i| i * 1000).collect();
match get_field_values_by_doc_ids_zero_copy("user.id", &doc_ids, arrow_file_path) {
    Ok((results, stats)) => {
        println!("Found {} unique values", results.len());
        stats.print_benchmark();
    }
    Err(e) => eprintln!("Error: {}", e),
}

// Example: Calculate numeric statistics
match get_numeric_stats_by_doc_ids_zero_copy::<i64>("payload_size", &doc_ids, arrow_file_path) {
    Ok((stats_result, query_stats)) => {
        println!("Sum: {}, Min: {}, Max: {}", 
            stats_result.sum, stats_result.min, stats_result.max);
        query_stats.print_benchmark();
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

## Comparison with Other Approaches

This implementation demonstrates several advantages over traditional approaches:

1. **Memory Efficiency**: Zero-copy processing minimizes memory usage
2. **Speed**: Direct memory access provides faster query execution
3. **Flexibility**: Complex nested data structures are fully supported
4. **Incremental Processing**: Efficient batch-by-batch processing of large datasets

## Requirements

* Rust (latest stable)
* Dependencies:  
   * arrow  
   * arrow-ipc  
   * arrow-buffer  
   * memmap2  
   * bytes 