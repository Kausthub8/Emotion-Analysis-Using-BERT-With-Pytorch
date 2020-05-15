from tqdm.notebook import tqdm 
      
for epoch in tqdm(range(1, epochs+1)):



  model.train()

  loss_train_total = 0 
  progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False) 
  for batch in progress_bar:

    model.zero_grad()
 
    batch = tuple(b.to(device) for b in batch)
 
    inputs = {'input_ids':   batch[0],
          'attention_mask': batch[1],
          'labels':         batch[2],
         }
  outputs = model(**inputs)
 
  loss=outputs[0]
  loss_train_total += loss.item()
  loss.backward()
 
  torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
 
  optimizer.step()
  scheduler.step()
 
  progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
 
 
 
  tqdm.write('\n Epoch {epoch}')
 
  loss_train_avg = loss_train_total/len(dataloader_train)
 
  tqdm.write(f'Training Loss: {loss_train_avg}')
 
  val_loss, predictions, true_vals = evaluate(dataloader_val)
  val_f1 = f1_score_func(predictions, true_vals)
  tqdm.write(f'Vlidation loss: {val_loss}')
  tqdm.write(f'F1 Score (weighted): {val_f1}')
