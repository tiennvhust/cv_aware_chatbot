#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 12:45:45 2025

@author: tienn
"""
import json
import pickle
import os
import argparse
from cryptography.fernet import Fernet
from pathlib import Path
import toml

class SecureDataTool:
    def __init__(self, key=None):
        """
        Initialize the tool.
        If a key is provided, use it. Otherwise, generate a new one.
        """
        if key:
            self.key = key
        else:
            print("No key provided. Generating a new encryption key...")
            self.key = Fernet.generate_key()
            print(f"**IMPORTANT** Save this key to decrypt later: {self.key.decode()}")
        
        self.cipher = Fernet(self.key)

    def json_to_encrypted_pickle(self, json_filepath, pickle_filepath):
        """
        Loads a JSON file, encrypts the data, and saves it as a pickle file.
        """
        if not os.path.exists(json_filepath):
            raise FileNotFoundError(f"The file {json_filepath} does not exist.")

        # 1. Load the JSON data
        with open(json_filepath, 'r') as f:
            data = json.load(f)

        # 2. Convert data to a JSON string, then to bytes
        json_bytes = json.dumps(data).encode('utf-8')

        # 3. Encrypt the bytes
        encrypted_data = self.cipher.encrypt(json_bytes)

        # 4. Pickle the encrypted data and write to file
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(encrypted_data, f)
        
        print(f"Success! Encrypted data saved to '{pickle_filepath}'")

    def load_encrypted_pickle(self, pickle_filepath):
        """
        Loads a pickle file, decrypts the content, and returns the JSON object (dict/list).
        """
        if not os.path.exists(pickle_filepath):
            raise FileNotFoundError(f"The file {pickle_filepath} does not exist.")

        # 1. Load the encrypted bytes from the pickle file
        with open(pickle_filepath, 'rb') as f:
            encrypted_data = pickle.load(f)

        # 2. Decrypt the data
        try:
            decrypted_bytes = self.cipher.decrypt(encrypted_data)
        except Exception as e:
            raise ValueError("Decryption failed. Invalid Key or Corrupted Data.") from e

        # 3. Convert back to JSON object
        json_data = json.loads(decrypted_bytes.decode('utf-8'))
        
        return json_data

def load_secret_key(key_path:str=".streamlit/secrets.toml") -> str:
    key = toml.load(key_path)
    return key["encode_key"]

# --- Example Usage ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, default='cv_atomic_db.json')
    args = parser.parse_args()
    file_name = Path(args.input_path).stem
    file_dir = os.path.dirname(file_name)
    # 2. Initialize the tool (generates a fresh key)
    # In a real scenario, store this key safely!
    encode_key = load_secret_key()
    tool = SecureDataTool(encode_key)
    
    # 3. Convert JSON to Encrypted Pickle
    tool.json_to_encrypted_pickle(args.input_path, os.path.join(file_dir, f"{file_name}.pkl"))

    # 4. Load it back to prove it works
    print("\n--- Attempting to reload data ---")
    restored_data = tool.load_encrypted_pickle(os.path.join(file_dir, f"{file_name}.pkl"))
    
    print("Restored Data:", restored_data)
    
    # Cleanup dummy file
    # os.remove("source_data.json")