#!/usr/bin/env python3
import json


class JSON_buffered_reader(object):
    def __init__(self, filename: str, serialization_array_depth: int = 1):
        self.filename = filename
        self.n = 0
        self.serialization_array_depth = serialization_array_depth

        self.array_d = 0
        self.object_d = 0

        self.file_obj = None
        self.buffer = r''

        self.json_queue = []

        self.eof = False

        self._open()

    def _open(self):
        self.n = 0
        self.array_d = 0
        self.object_d = 0
        self.file_obj = open(self.filename, 'r')

    def _close(self):
        self.file_obj.close()

    def _process_next_line(self):
        line = self.file_obj.readline()

        if len(line) == 0:
            self.eof = True
            self._close()
            return
        
        prev_c = ''

        in_quotes = False

        for c in line:
            
            if prev_c == '\\':
                self.buffer += c
                prev_c = '' if c == '\\' else c
                continue
            elif c == '\n':
                continue
            if not in_quotes:
                if c == '[':
                    if self.array_d >= self.serialization_array_depth:
                        self.buffer += c
                    self.array_d += 1
                elif c == ']':
                    if self.array_d >= self.serialization_array_depth:
                        self.buffer += c
                    self.array_d -= 1
                elif c == '{':
                    self.object_d += 1
                    self.buffer += c
                elif c == '}':
                    self.object_d -= 1
                    self.buffer += c
                elif c == '"':
                    in_quotes = True
                    self.buffer += c
                elif c == ',':
                    if not(self.array_d == self.serialization_array_depth and self.object_d == 0):
                        self.buffer += c
                elif c == '\n' or c == ' ' or c == '\t':
                    continue
                else:
                    self.buffer += c
                    
            else:
                if c == '"':
                    in_quotes = False
                self.buffer += c

            assert self.object_d >= 0
            assert self.array_d >= 0
            
            prev_c = c

            if self.object_d == 0:
                if self.array_d == self.serialization_array_depth and len(self.buffer) > 0:
                    self.json_queue.append(self.buffer)
                    self.buffer = r''

    def __iter__(self):
        return self

    def __next__(self):
        while len(self.json_queue) == 0:
            if self.eof:
                raise StopIteration
            self._process_next_line()
        tmp = self.json_queue.pop(0)
        return json.loads(tmp)
