#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem> 

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    // The program expects exactly two command-line arguments:
    // argv[0] is the program name itself.
    // argv[1] is the path to the dataset folder.
    // argv[2] is the path for the output CSV file.
    if (argc != 3) {
        // Print a usage message to the console if arguments are wrong.
        std::cerr << "Usage: " << argv[0] << " <path_to_dataset_folder> <output_csv_file>" << std::endl;
        std::cerr << "Example: " << argv[0] << " ./Dataset ./labels.csv" << std::endl;
        return 1; // Return an error code
    }

    fs::path dataset_path(argv[1]);
    std::string output_csv_path(argv[2]);

    std::ofstream csv_file(output_csv_path);
    if (!csv_file.is_open()) {
        std::cerr << "Error: Could not open output file for writing: " << output_csv_path << std::endl;
        return 1; 
    }

    std::cout << "INFO: Output CSV file opened successfully at: " << output_csv_path << std::endl;

    csv_file << "image,label\n";

    try {
        std::cout << "INFO: Processing dataset directory: " << dataset_path.string() << std::endl;
        for (const auto& class_entry : fs::directory_iterator(dataset_path)) {
            // Check if the entry is a directory (e.g., "class1_folder")
            if (class_entry.is_directory()) {
                const fs::path& class_path = class_entry.path();
                // The label is the name of the folder
                std::string label = class_path.filename().string();

                std::cout << "  -> Found class: " << label << std::endl;

                // iterate through all files inside this class directory
                for (const auto& image_entry : fs::directory_iterator(class_path)) {
                    // Check if the entry is a regular file (i.e., not a folder)
                    if (image_entry.is_regular_file()) {
                        // The image name is the name of the file
                        std::string image_name = image_entry.path().filename().string();
                        
                        // Write the "image_name,label" pair to the CSV file
                        csv_file << image_name << "," << label << "\n";
                    }
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem Error: " << e.what() << std::endl;
        return 1;
    }
    csv_file.close();
    std::cout << "\nSUCCESS: CSV file has been generated successfully." << std::endl;

    return 0; // Success
}
