logging_level = [0, 1, 2]


# todo: suppress commandline output

def logging_config_dict(level=0, output_filename='./example.log'):
    level = min(level, len(logging_level) - 1)
    console_level = ['WARNING', 'INFO', 'DEBUG', ][level]
    file_level = ['DEBUG', 'DEBUG', 'DEBUG'][level]
    LOGGING = {
        'version': 1,
        'disable_existing_loggers': True,
        'formatters': {
            'default': {
                'format': '[%(levelname)s - %(asctime)s - %(module)s] %(message)s'
            },
        },
        'handlers': {
            'console': {
                'level': console_level,
                'class': 'logging.StreamHandler',
                'formatter': 'default'
            },
            'file': {
                'level': file_level,
                'class': 'logging.FileHandler',
                'filename': output_filename,
                'mode': 'a',
                'formatter': 'default',
            },
        },
        'loggers': {
            '': {  # root logger
                'handlers': ['console', 'file'],
                'level': level,
            },
        }
    }
    return LOGGING


class EvalLog:
    def __init__(self, step, epoch, data, method, value, *args, **kwargs):
        self.values = [step, epoch, data, method, value]

    def __str__(self, ):
        return "$EVAL " + " ".join(str(i) for i in self.values)

    @staticmethod
    def load(eval_str):
        info = eval_str.split()[1:]
        return [int(info[0]), int(info[1]), info[2], info[3], float(info[4]), ]


def ArgLog(key, value):
    return "$ARGUMENT {}:{}".format(key, value)
