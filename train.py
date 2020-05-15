def evaluate(dataloader_val):

  model.eval()

  loss_val_total = 0
  predictions, true_vals = [], []

  for batch in dataloader_val:

    batch = tuple(b.to(device) for b in batch)

    inputs = {'input_ids':   batch[0],
              'attention_mask': batch[1],
              'labels':         batch[2],
             }
  with torch.no_grad():
      outputs=model(**inputs)


      loss=outputs[0]
      logits=outputs[1]
      loss_val_total += loss.item()


      logits = logits.detach().cpu().numpy()
      label_ids = inputs['labels'].cpu().numpy()
      predictions.append(logits)
      true_vals.append(label_ids)

      loss_val_avg = loss_val_total/len(dataloader_val)


      predictions = np.concatenate(predictions, axis=0)
      true_vals = np.concatenate(true_vals, axis=0)

      return loss_val_avg, predictions, true_vals
