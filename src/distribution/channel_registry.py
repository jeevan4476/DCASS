# src/distribution/channel_registry.py

from .console_channel import ConsoleChannel
from .local_folder_channel import LocalFolderChannel

def get_available_channels():
    return {
        "console": ConsoleChannel(),
        "local": LocalFolderChannel()
    }
