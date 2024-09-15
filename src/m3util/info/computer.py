def parse_system_info(self, file_path):
        system_info = {"System Information": {}}
        gpu_num = 0
        GPU = False
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith("GPU: NVIDIA"):
                    gpu_num += 1
                    GPU = True
                    gpu_key = f"GPU Information_{gpu_num}"
                    system_info["System Information"][f"{gpu_key}"] = {} 
                 
                if GPU:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        system_info["System Information"][f"{gpu_key}"][key.strip()
                                                        ] = value.strip()
                else:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        system_info["System Information"][key.strip()
                                                        ] = value.strip()

        self.system_info_ = system_info