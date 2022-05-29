import sys

def progress_bar(current_iter : int, total_iter : int, finished : bool = False):
        bar_len = 60
        perc = current_iter / float(total_iter)
        filled_len = int(round(bar_len * perc))

        percents = round(100.0 * perc, 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)
        finisher = '\n' if finished else '\r'

        sys.stdout.write('[%s] %s%s ...%s/%s rows%s' % (bar, percents, '%', current_iter, total_iter, finisher))
        sys.stdout.flush()