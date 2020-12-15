import os


def data_dir(*args):
    root_dir = os.path.expanduser(
        os.path.expandvars(os.environ.get("GUD_DATA", "~/gud_data"))
    )
    os.makedirs(root_dir, exist_ok=True)
    return os.path.join(root_dir, *args)
