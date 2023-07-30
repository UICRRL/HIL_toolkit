# Raspeberry Pi Instructions

## Install LSL on Raspberry Pi
1. Install git.
```bash
sudo apt-get install git
```

2. Check the status of the installation script and update to executable if necessary.
```bash
chmod +x setup_liblsl_rasp.sh
```

3. Run the installation script.
```bash
./setup_liblsl_rasp.sh
```

This script will clone the liblsl and install lsl. It will also set the `liblsl.so` path to the `PYLSL_LIB` environment variable for your Raspberry Pi.