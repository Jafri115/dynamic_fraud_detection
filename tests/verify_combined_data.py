import os
import numpy as np
import pandas as pd

# Load original processed user-level data
train_df = pd.read_pickle("data/processed/wiki/user_edits_train.pkl")
val_df = pd.read_pickle("data/processed/wiki/user_edits_val.pkl")

# Load tabular and sequence files
x_train = np.load("data/processed/combined/x_train.npy")
x_val = np.load("data/processed/combined/x_val.npy")

train_labels = np.load("data/processed/combined/wiki_train_labels.npy")
val_labels = np.load("data/processed/combined/wiki_val_labels.npy")

train_seq = np.load("data/processed/combined/wiki_train_sequence.npz", allow_pickle=True)
val_seq = np.load("data/processed/combined/wiki_val_sequence.npz", allow_pickle=True)

def verify_alignment(df, x_tab, y_label, seq_data, split_name):
    assert len(df) == x_tab.shape[0] == y_label.shape[0] == seq_data['event_seq'].shape[0], \
        f"[{split_name}] Row count mismatch: df={len(df)}, x={x_tab.shape[0]}, y={y_label.shape[0]}, seq={seq_data['event_seq'].shape[0]}"

    print(f"[{split_name}] All data lengths match")

    # Spot-check alignment for a few samples
    for i in [0, len(df)//2, len(df)-1]:
        row = df.iloc[i]
        assert row['label'] == y_label[i], f"[{split_name}] Label mismatch at index {i}: df={row['label']} vs label={y_label[i]}"
        
        seq_len = len(seq_data['event_seq'][i])
        orig_seq_len = len(row['edit_sequence'])
        
        # Check if sequences match (allowing for potential padding tokens)
        if seq_len != orig_seq_len:
            print(f"[{split_name}] Warning: Sequence length mismatch at index {i}: {seq_len} vs {orig_seq_len}")
            
            # Check if the difference is due to padding/special tokens
            if seq_len == orig_seq_len + 1:
                # Check if last element is an EOS token (should be [273040] - the no_cat value)
                last_elem = seq_data['event_seq'][i][-1]
                if isinstance(last_elem, list) and len(last_elem) == 1 and last_elem[0] == 273040:
                    print(f"[{split_name}] Detected expected EOS token: {last_elem}")
                    # Compare sequences excluding the last element
                    if seq_data['event_seq'][i][:-1] == row['edit_sequence']:
                        print(f"[{split_name}] Sequences match after excluding EOS token")
                    else:
                        print(f"[{split_name}] Error: Sequences don't match even after excluding EOS token")
                        raise AssertionError(f"[{split_name}] Sequence content mismatch at index {i}")
                else:
                    print(f"[{split_name}] Unexpected EOS token: {last_elem} (expected [273040])")
                    # For backward compatibility, also accept [299] if data was processed with old settings
                    if isinstance(last_elem, list) and len(last_elem) == 1 and last_elem[0] == 299:
                        print(f"[{split_name}] Found old EOS token [299] - data needs reprocessing with new settings")
                        if seq_data['event_seq'][i][:-1] == row['edit_sequence']:
                            print(f"[{split_name}] Sequences match (but using old EOS token)")
                        else:
                            raise AssertionError(f"[{split_name}] Sequence content mismatch at index {i}")
                    else:
                        raise AssertionError(f"[{split_name}] Unexpected EOS token at index {i}: expected [273040], got {last_elem}")
            else:
                raise AssertionError(f"[{split_name}] Sequence length mismatch at index {i}: {seq_len} vs {orig_seq_len}")
        else:
            # Exact length match - verify content
            if seq_data['event_seq'][i] != row['edit_sequence']:
                raise AssertionError(f"[{split_name}] Sequence content mismatch at index {i}")

        print(f"[{split_name}] Row {i} check passed | Label: {y_label[i]} | Seq len: {seq_len} | Total edits: {row['total_edits']}")

# Run verifications
verify_alignment(train_df, x_train, train_labels, train_seq, "TRAIN")
verify_alignment(val_df, x_val, val_labels, val_seq, "VAL")

print("[OK] All combined data alignments verified successfully.")
