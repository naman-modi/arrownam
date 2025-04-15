use arrow::ipc::reader::FileReader;
use arrow::ipc::writer::FileWriter;
use arrow::record_batch::RecordBatch;
use memmap2::Mmap;
use std::fs::File;
use std::io::Cursor;

/// Writes a record batch to an Arrow IPC file
pub fn write_batch_ipc(path: &str, batch: &RecordBatch) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = FileWriter::try_new(file, &batch.schema())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    writer
        .write(batch)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    writer
        .finish()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    Ok(())
}

/// Reads record batches from an Arrow IPC file
pub fn read_batches_ipc(path: &str) -> arrow::error::Result<Vec<RecordBatch>> {
    let file = File::open(path)?;
    let mmap_new = unsafe { Mmap::map(&file)? };
    let cursor = Cursor::new(&mmap_new[..]);
    let reader = FileReader::try_new(Box::new(cursor), None)?;
    reader.collect()
}
