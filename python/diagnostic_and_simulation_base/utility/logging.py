import datetime
import logging
import sys

# Set the logging level.
# Adjusting logging level changes what events are retained in the log.
# Options from least to most verbose are:
# * logging.CRITICAL
# * logging.ERROR
# * logging.WARNING
# * logging.INFO
# * logging.DEBUG
# * logging.NOTSET
logger = logging.getLogger("diagnostics_analysis_base")
logger.setLevel(level=logging.DEBUG)


class EmitLogToListHandler(logging.Handler):
    """Custom handler to "emit" the log records into a list"""

    def __init__(self, log_records: list[logging.LogRecord]) -> None:
        self.log_records = log_records
        logging.Handler.__init__(self)

    def emit(self, record: logging.LogRecord) -> None:
        self.log_records.append(record)


class CustomFormatter(logging.Formatter):
    """Custom format"""

    def format(self, record: logging.LogRecord) -> str:
        date_time = datetime.datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        filename_and_lineno = f"{record.filename}:{record.lineno}"
        return f"{date_time}  |  {record.name}  |  {filename_and_lineno:<37}  |  {record.levelname}  |  {record.msg}"


def format_logs(log_records: list[logging.LogRecord]) -> str:
    """This converts a list of logging records into a string.
    This is used to save the logging to MDSplus.
    """
    formatter = CustomFormatter()
    str_output = ""
    for log_record in log_records:
        formatted_log = formatter.format(log_record)
        str_output += formatted_log + "\n"
    return str_output[0:-1]


# def create_logger(log_name: str):

#     # Create an empty list where logs will be stored
#     log_records = []

#     # Create the logger
#     logger = logging.getLogger(log_name)

#     # Set the logging level. Adjusting this will change what events are retained in
#     # the log. Options from least to most verbose are:
#     # * logging.CRITICAL
#     # * logging.ERROR
#     # * logging.WARNING
#     # * logging.INFO
#     # * logging.DEBUG
#     # * logging.NOTSET
#     logger.setLevel(level=logging.DEBUG)

#     # Create and add handler to emit the logs to "stadnard out",
#     # i.e. printing to the terminal (standard behaviour)
#     handler_to_standard_out = logging.StreamHandler(sys.stdout)
#     handler_to_standard_out.setFormatter(CustomFormatter())
#     logger.addHandler(handler_to_standard_out)

#     # Create and add handler to emit the log "records" into the `log_records` list
#     handler_to_list = EmitLogToListHandler(log_records)
#     logger.addHandler(handler_to_list)

#     return logger, log_records
