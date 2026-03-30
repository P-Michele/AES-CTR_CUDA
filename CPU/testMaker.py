import os
import pathlib


sizes_gb = [1, 2, 4, 8]

for size in sizes_gb:
    file_path = pathlib.Path(__file__).parent.resolve() / f"test_{size}GB.bin"
    print(f"Creating {size}GB file...")
    
    with open(file_path, "wb") as f:
        for i in range(size):
            # Write 1GB chunk to avoid RAM overload
            f.write(os.urandom(1024 * 1024 * 1024))
            print(f"Written GB {i+1} of {size}")
            
    print(f"Completed: {file_path}\n")