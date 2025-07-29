from datetime import datetime
from enum import IntFlag
from operator import ior
from functools import reduce
from pathlib import Path
from sys import stdout, exc_info
from typing import TextIO

import click
import psutil


class LOG(IntFlag):
    SILENT = 0     # ("SILENT", 0, "black")
    ERROR = 1     # ("ERROR", 1, "red")
    STATUS = 2     # ("STATUS", 2, "green")
    TIME = 4     # ("TIME", 3, "cyan")
    WARN = 8      # ("WARN", 4, "yellow")
    INFO = 16     # ("INFO", 5, "white")
    DEBUG = 32     # ("INFO", 5, "magenta")

    def __init__(self, log_level):
        self.log_level = log_level
        self.__colors = {0: "black", 1: "red", 2: "green", 4: "cyan", 8: "yellow", 16: "white", 32: "magenta"}

    def __str__(self):
        return str(self.name)

    @property
    def color(self):
        return self.__colors[self.value]


class Logger:
    def __init__(self, log_screen=None, log_files=None):
        """Initializes new logger, with starting time and log target locations.

            :param log_screen: Dictionary showing what values are written to stdout and stderr, based on LOG flags.
            :param log_files: Dictionary of log file with the statuses they will write to.
        """

        self.script_start = datetime.now()
        self.__attached_funcs = []
        if log_screen is not None:
            self.__log_screen = log_screen
        else:
            self.__log_screen = {
                "stdout": LOG.STATUS | LOG.TIME | LOG.WARN | LOG.INFO,
                "stderr": LOG.ERROR
            }
        if log_files is not None:
            self.__logs = log_files
        else:
            self.__logs = {
                "general": (Path(f"imaging_log_{str(self.script_start).replace(':', '')}.txt"),
                            ~(LOG.INFO or LOG.DEBUG))
            }
        self.__pid = psutil.Process().pid

    def set_logdir(self, logdir):
        self.__logs["general"] = (Path(logdir, f"imaging_log_{str(self.script_start).replace(':', '')}.txt"),
                                  self.__logs["general"][1])
        Path(logdir).mkdir(parents=True, exist_ok=True)

    def header(self, out=None):
        """Writes explanatory header for the logging format.

            :param out: Stream to write to.  If not passed, will write to default log screens and files.
        """
        header_message = f'{"TYPE":6}|{"STEP":20}|   TIMESTAMP   |MEM USAGE|MEM FREE | STATEMENT '
        self.__special_write(header_message, out)

    def footer(self, out=None, error=None):
        step = "COMPLETED" if error is None else "ERRORED"
        statement = "" if error is None else f"{error}"

        footer_message = self.__log_message(step, statement)
        self.__special_write(footer_message, out)

    def __log_message(self, step: str, statement: str = '', log_level: LOG = LOG.TIME):
        styled_type = click.style(f'{log_level.name:6}', log_level.color)
        message = (f'{styled_type}'
                   f'|{step[:20]:20}'
                   f'|{str(datetime.now() - self.script_start).zfill(15)}'
                   f'|{psutil.Process().memory_info().vms // 1024 ** 3:07.2f}GB'
                   f'|{psutil.virtual_memory().available // 1024 ** 3:07.2f}GB'
                   f'|"{statement}"')
        return message

    def __special_write(self, message, out=None):
        if out is not None:
            click.echo(message, file=out)
        else:
            if "stdout" in self.__log_screen and self.__log_screen["stdout"] is not None:
                click.echo(message)
            elif "stderr" in self.__log_screen and self.__log_screen["stderr"] is not None:
                click.echo(message, err=True)

            for _name, (log_path, _log_flag) in self.__logs.items():
                with open(log_path, "a") as handle:
                    click.echo(message, file=handle)

    def __multi_write(self, message, log_level):
        if log_level & reduce(ior, self.__log_screen.values()):
            click.echo(message, err=log_level & self.__log_screen["stderr"])

        for _name, (log_path, log_flag) in self.__logs.items():
            if log_level & log_flag:
                with open(log_path, "a") as handle:
                    click.echo(message, file=handle)

    def write(self, step: str, statement: str = '', log_level: LOG = LOG.TIME, out=None, write_tb=False):
        """Writes formatted log information with timing and memory usage.

            Data is written in the format

            TYPE  :       STEP         |   TIMESTAMP   |MEM USE |MEM FREE| STATEMENT

            With timestamp being time since script start, and memory in MB.

            Logs can be written into two places, based on thresholding (see set_logger):
                (1) stdout/stderr (base logging).
                (2) Specified openable streams.

            :param step: Step the process is currently at.  (Required)
            :param statement: Any (string-formattable) value to write to log.
            :param log_level: Type (based on DEBUG) of log to write, determining color and whether to write.
            :param out: Stream to write to, instead of defined logging streams.
            :param write_tb: Whether to write an (unformatted) traceback dump to log(s) in addition to logging
                             statement.
            """
        for func in self.__attached_funcs:
            func(step, self.__pid)

        if write_tb:
            _, _, last_traceback = exc_info()
            statement += f'-tb-{last_traceback}'

        log_line = self.__log_message(step, statement, log_level)

        if out is not None:
            click.echo(log_line, file=out, err=(log_level == LOG.ERROR))
        else:
            self.__multi_write(log_line, log_level)

    def confirm(self, step: str, statement: str = '', log_level: LOG = LOG.TIME, out: TextIO = stdout):
        """Writes a confirmation prompt in matching format.

            :param step: Step the process is currently at.  (Required)
            :param statement: Any (string-formattable) value to write to log.
            :param log_level: Type (based on DEBUG) of log to write, determining color and whether to write.
            :param out: Stream to write to, if not stdout (default).
        """
        for func in self.__attached_funcs:
            func(step, self.__pid)
        return click.confirm(self.__log_message(step, statement, log_level), err=(log_level == LOG.ERROR))

    def prompt(self, step: str, statement: str = '', log_level: LOG = LOG.TIME, out: TextIO = stdout, default=None):
        """Writes a general prompt in matching format.

            :param step: Step the process is currently at.  (Required)
            :param statement: Any (string-formattable) value to write to log.
            :param log_level: Type (based on DEBUG) of log to write, determining color and whether to write.
            :param out: Stream to write to, if not stdout (default).
            :param default: Default value of the prompt.
        """
        for func in self.__attached_funcs:
            func(step, self.__pid)
        return click.prompt(self.__log_message(step, statement, log_level), err=(log_level == LOG.ERROR),
                            default=default)

    def attach_func(self, func: callable):
        """Attaches a function or other callable object to be called (with the PID) each time something is logged.

            :param func: Any callable object.
        """
        if func not in self.__attached_funcs:
            self.__attached_funcs.append(func)

    def dump(self, statement: str, log_level: LOG = LOG.INFO, out=None, use_color=False):
        """Writes a line without the format or any metadata (type, step, timestamp, memory usage).
            Attached functions aren't run.

            :param statement: Any (string-formattable) value to write to log.
            :param log_level: Type (based on DEBUG) of log to write, determining color and whether to write.
            :param out: Stream to write to, instead of defined logging streams.
            :param use_color: Whether to write the entire statement with the log color type.
        """
        if use_color:
            statement = click.style(statement, log_level.color)

        if out is not None:
            click.echo(statement, file=out, err=(log_level == LOG.ERROR))
        else:
            self.__multi_write(statement, log_level)


log = Logger()
