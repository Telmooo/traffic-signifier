import sys

import torch
from torch import nn
from tqdm import tqdm

def epoch_iter(dataloader, model, loss_fn, metric_scorer, device : str, optimizer=None, is_train=True):
    if is_train:
      assert optimizer is not None, "When training, please provide an optimizer."
      
    num_batches = len(dataloader)

    if is_train:
      model.train() # put model in train mode
    else:
      model.eval()

    total_loss = 0.0
    preds = []
    labels = []

    with torch.set_grad_enabled(is_train):
      for _batch, (X, y) in enumerate(tqdm(dataloader)):
          X, y = X.to(device), y.to(device)

          # Compute prediction error
          pred = model(X)
          loss = loss_fn(pred, y)

          if is_train:
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

          # Save training metrics
          total_loss += loss.item() # IMPORTANT: call .item() to obtain the value of the loss WITHOUT the computational graph attached

          probs = nn.functional.softmax(pred, dim=1)
          final_pred = torch.argmax(probs, dim=1)
          preds.extend(final_pred.cpu().numpy())
          labels.extend(y.cpu().numpy())

    return total_loss / num_batches, metric_scorer(labels, preds)

def progress_bar(current_iter : int, total_iter : int, finished : bool = False):
        bar_len = 60
        perc = current_iter / float(total_iter)
        filled_len = int(round(bar_len * perc))

        percents = round(100.0 * perc, 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)
        finisher = '\n' if finished else '\r'

        sys.stdout.write('[%s] %s%s ...%s/%s rows%s' % (bar, percents, '%', current_iter, total_iter, finisher))
        sys.stdout.flush()