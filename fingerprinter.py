from dejavu import Dejavu

config = {
    "database": {
        "host": "127.0.0.1",
        "user": "root",  # Your MySQL username, 'root' is the default
        "passwd": "debang",  # Your MySQL password
        "db": "dejavu",  # Your database name
    }
}

djv = Dejavu(config)

djv.fingerprint_directory('mp3', ['mp3', 'wav','flac'])


