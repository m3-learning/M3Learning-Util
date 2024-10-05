# ## TODO: might want to merge with the class where this came from. 

# def parse_system_info(self, file_path):
#     """
#     Parses system information from a given file and stores it in the instance variable `system_info_`.

#     Args:
#         file_path (str): The path to the file containing system information.

#     The method reads the file line by line and extracts key-value pairs of system information.
#     It specifically looks for GPU information and organizes it under separate keys for each GPU found.
#     """
#     system_info = {"System Information": {}}
#     gpu_num = 0
#     GPU = False

#     # Open the file for reading
#     with open(file_path, 'r') as file:
#         for line in file:
#             # Check if the line indicates the start of GPU information
#             if line.startswith("GPU: NVIDIA"):
#                 gpu_num += 1
#                 GPU = True
#                 gpu_key = f"GPU Information_{gpu_num}"
#                 system_info["System Information"][f"{gpu_key}"] = {}

#             # If currently parsing GPU information
#             if GPU:
#                 if ":" in line:
#                     key, value = line.split(":", 1)
#                     system_info["System Information"][f"{gpu_key}"][key.strip()] = value.strip()
#             else:
#                 # General system information
#                 if ":" in line:
#                     key, value = line.split(":", 1)
#                     system_info["System Information"][key.strip()] = value.strip()

#     # Store the parsed information in the instance variable
#     self.system_info_ = system_info