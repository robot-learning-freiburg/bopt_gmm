class CSVLogger(object):
    def __init__(self, path, fields):
        path = path if path[-4:] == '.csv' else f'{path}.csv'
        self._fields = sorted(set(fields))
        self._file   = open(path, 'w')

        # Write Header
        self._file.write(f'{",".join(self._fields)}\n')

    def __del__(self):
        if self._file is not None:
            self._file.close()

    def log(self, v_dict):
        if self._file is None:
            raise Exception('Logger file is closed')

        values = [str(v_dict[f]) for f in self._fields]

        self._file.write(f'{",".join(values)}\n')
        self._file.flush()
    
    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
