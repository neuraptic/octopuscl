# Datasets for OctopusCL

Datasets are composed of multiple files and folders, which must be placed in the root of the dataset directory.

## 1. `schema.json` file

The dataset schema provides detailed information about the dataset itself. The schema must be defined in the 
`schema.json` JSON file, and must include the following fields:

- `name`: A unique identifier for the dataset.
- `description`: A brief explanation of the dataset and its purpose.
- `inputs`: Data fields that serve as inputs.
- `outputs`: Data fields that serve as outputs.
- `metadata`: Data fields providing additional information that is not used for inference or training.

Data fields are called *elements* in OctopusCL. An element is defined by the following properties:

- `name`: Name of the element.
- `description`: Description of the element.
- `required`: All examples must have a value for the element
- `nullable`: Allow null values
- `multi_value`: Allow multiple values. Three formats supported: "unordered", "ordered", and "time_series" 
  (same as "ordered", but in this case the index of a value represents a time step). 
  If no format is provided, the element will not allow multiple values.

Example structure of `schema.json`:
```
{
    "name": "test_dataset",
    "description": "A test dataset",
    "inputs": [
        {
            "name": "input_1",
            "type": "boolean",
            "required": true,
            "nullable": false
        },
        {
            "name": "input_2",
            "type": "integer",
            "required": false,
            "nullable": true
        },
        {
            "name": "input_3",
            "type": "document_file",
            "required": false,
            "nullable": true
        }
    ],
    "outputs": [
        {
            "name": "output_1",
            "type": "text",
            "required": true,
            "nullable": false
        }
    ],
    "metadata": [
        {
            "name": "metadata_1",
            "type": "datetime",
            "required": true,
            "nullable": false
        }
    ]
}
```

## 2. `examples.csv` or `examples.db` file

Examples must be stored in a CSV file (`examples.csv`) or an SQLite database (`examples.db`). These two formats are 
mutually exclusive, meaning that only one format should be used, not both.

Format requirements:

- `examples.csv`
    - Each row should correspond to a unique example.
    - Columns should represent inputs, outputs, and metadata fields.
    - The first column must be the example's Universally Unique Identifier (UUID), adhering to the UUID4 standard. You 
      can easily generate one using Python with the `uuid.uuid4()` function.
- `examples.db`
    - The database should contain only one table named `examples`.
    - The `examples` table should have the same structure as the one described for the CSV format.

Example structure of `examples.csv`:

```
id,input_1,input_2,input_3,output_1,metadata_1
f31ce299-fe2c-4a03-97f4-13d3930ebca2,true,123,file_1.txt,value_output_1,2023-08-29T19:04:25
...
```

## 3. `files` folder (optional)

Examples may reference files, such as images, documents, or any other type of file. These files should ideally be 
placed in the `files` folder to centralize artifacts and simplify dataset management. In cases where files need to be 
stored in other locations, ensure that the file paths in the examples are absolute. This is particularly useful when 
multiple datasets share the same files, as it helps avoid duplication and conserves disk space.

If the examples use relative file paths, it is assumed the files are located in the `files` folder.

## 4. `splits` folder (optional)

Examples can be split into different partitions for training, validation, and testing purposes. In continual learning 
scenarios, examples can be further split into experiences.

To reproduce the same splits across different experiments, the splits can be pre-defined in the `splits` directory 
using the following structure:

- `experience_0/`: Contains the splits belonging to experience 0.
    - `partition_0`: Contains the splits belonging to partition 0.
        - `training.txt`: Contains the indices of the examples in the training set.
        - `test.txt`: Contains the indices of the examples in the test set.
        - `validation.txt` (optional): Contains the indices of the examples in the validation set.
    - `partition_1`: Contains the splits belonging to partition 1.
    - ...
- `experience_1/`: Contains the splits belonging to experience 1.
    - ...
- ...
