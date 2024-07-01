# Use the existing Docker image as the base image
FROM eedddyyyybae/monodepth2_pytorch_sit:latest

# Set up environment variables (if needed)
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all
ENV QT_X11_NO_MITSHM 1

# Copy any necessary files from the host to the container (if needed)
# COPY /path/on/host /path/in/container

# Set up any additional configurations (if needed)

# Specify the command to run on container start
CMD ["/bin/bash"]
