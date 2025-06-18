import itertools
import torch 
from torch.utils.data import IterableDataset, get_worker_info

class PreprocessedIterableDataset_v2(IterableDataset):
    def __init__(self, data, tokenizer, batch_size, max_length, start_tokenizing_idx):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        
        self.start_tokenizing_idx = start_tokenizing_idx
        self.k = 0

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            # If no worker_info is provided, we are not using DataLoader workers, so yield all data
            iter_data = iter(self.data)
        else:
            # If using DataLoader workers, yield a subset of the data for this worker
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            iter_data = self.data
            #iter_data = itertools.islice(self.data, worker_id, None, num_workers)

        batch = []
        

        
        for example in iter_data:
            if self.k > self.start_tokenizing_idx :
                tokenized_example = self.tokenizer(
                    example["text"],
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                batch.append(tokenized_example)
            else:
                batch.append(0)
                

            if len(batch) == self.batch_size:
                yield self._format_batch(batch)
                batch = []
                
                self.k += 1

        if batch:
            yield self._format_batch(batch)

    def _format_batch(self, batch):
        if self.k > self.start_tokenizing_idx:
            input_ids = torch.stack([item["input_ids"].squeeze(0) for item in batch])
            attention_mask = torch.stack([item["attention_mask"].squeeze(0) for item in batch])
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        else:
            return 0
