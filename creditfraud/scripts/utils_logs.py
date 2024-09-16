import os
import gzip

def split_and_compress_log_file(file_path, num_segments=10, output_prefix="log_segment"):
    # Get the size of the file
    file_size = os.path.getsize(file_path)
    segment_size = file_size // num_segments
    
    with open(file_path, "r") as file:
        for segment_index in range(num_segments):
            segment = file.read(segment_size)
            if not segment:
                break
            output_file_path = f"{output_prefix}_{segment_index}.log.gz"
            with gzip.open(output_file_path, "wt") as output_file:
                output_file.write(segment)
            print(f"Segment {segment_index} written to {output_file_path}")

# Path to the detailed debug log file
log_file_path = "/home/smf19/creditfraud/detailed_debug.log"

# Split and compress the detailed debug log file into 10 segments
split_and_compress_log_file(log_file_path, num_segments=10)
