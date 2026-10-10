def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
	"""Given the path to an input corpus, run train a BPE tokenizer and
	output its vocabulary and merges.

	Args:
		input_path (str | os.PathLike): Path to BPE tokenizer training data.
		vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
		special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
			These strings will never be split into multiple tokens, and will always be
			kept as a single token. If these special tokens occur in the `input_path`,
			they are treated as any other string.

	Returns:
		tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
			vocab:
				The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
				to bytes (token bytes)
			merges:
				BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
				representing that <token1> was merged with <token2>.
				Merges are ordered by order of creation.
	"""
	def vocabulary(merges: list[tuple[bytes, bytes]], special_tokens: list[str]) -> dict[int, bytes]:
		
		vocab = {}
		token2id = {}
		
		# 1.initial byte vocabulary
		for i in range(256):
			vocab[i] = bytes([i])
		
		for idx, token in vocab.items():
			token2id[token] = idx
		
		next_id = 256
		
		# 2.vocabulary items produced from merging
		for token1, token2 in merges:
			new_token = token1 + token2
			if new_token not in token2id:
				vocab[next_id] = new_token
				token2id[new_token] = next_id
				next_id += 1
					
		# 3.any special tokens
		for item in special_tokens:
			item_bytes = item.encode("utf-8")
			if item_bytes in token2id:
				continue
			vocab[next_id] = item_bytes
			token2id[item_bytes] = next_id
			next_id += 1
		
		return vocab
	
	def pre_tokenize(chunk: str) -> dict[str, int]:
		count = {}
		pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
		
		for token in pat.findall(chunk):
			count[token] = count.get(token, 0) + 1
		return count
		
	def merge(freq_table: dict[tuple[bytes, ...], int], vocab_size: int, num_special: int) -> list[tuple[bytes, bytes]]:
		merges = []
		
		# 1.count adjacent pairs once, and remember which pairs each word has
		pairs = {}
		word_pairs = {}
		for k, v in freq_table.items():
			kp = []
			for i in range(len(k) - 1):
				pair = (k[i], k[i + 1])
				pairs[pair] = pairs.get(pair, 0) + v
				kp.append(pair)
			word_pairs[k] = kp
		
		while True:
			if not pairs:
				break
			
			# 2.pick most frequent, tie-break by lexicographically greater pair
			max_pair = max(pairs.items(), key = lambda kv: (kv[1], kv[0]))[0]
			merges.append(max_pair)
			
			# 3.remove the words that contain max_pair, undoing their pair counts
			affected = {}
			for k in list(word_pairs.keys()):
				if max_pair in word_pairs[k]:
					v = freq_table[k]
					affected[k] = v
					for pair in word_pairs[k]:
						pairs[pair] -= v
						if pairs[pair] <= 0:
							del pairs[pair]
					del word_pairs[k]
					del freq_table[k]
			
			# 4.rewrite those words, collapsing max_pair
			new_words = {}
			for k, v in affected.items():
				new_key = []
				i = 0
				while i < len(k):
					if i < len(k) - 1 and (k[i], k[i + 1]) == max_pair:
						new_key.append(k[i] + k[i + 1])
						i += 2
					else:
						new_key.append(k[i])
						i += 1
				t = tuple(new_key)
				new_words[t] = new_words.get(t, 0) + v
			
			# 5.put them back, merging with any existing word of the same form
			for t, v in new_words.items():
				if t in freq_table:
					for pair in word_pairs[t]:
						pairs[pair] -= freq_table[t]
						if pairs[pair] <= 0:
							del pairs[pair]
					freq_table[t] += v
				else:
					freq_table[t] = v
				tp = []
				for i in range(len(t) - 1):
					pair = (t[i], t[i + 1])
					pairs[pair] = pairs.get(pair, 0) + freq_table[t]
					tp.append(pair)
				word_pairs[t] = tp
			
			# 6.compute the vocabulary_size to decide whether stop
			if 256 + len(merges) + num_special >= vocab_size:
				break
		
		return merges
	
	def find_chunk_boundaries(file, desired_num_chunks, split_special_token):
		file.seek(0, os.SEEK_END)
		file_size = file.tell()
		file.seek(0)

		chunk_size = file_size // desired_num_chunks
		boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
		boundaries[-1] = file_size

		mini_chunk_size = 4096
		overlap = len(split_special_token) - 1

		for bi in range(1, len(boundaries) - 1):
			file.seek(boundaries[bi])
			while True:
				start = file.tell()
				mini_chunk = file.read(mini_chunk_size)
				if mini_chunk == b"":
					boundaries[bi] = file_size
					break
				found_at = mini_chunk.find(split_special_token)
				if found_at != -1:
					boundaries[bi] = start + found_at
					break
				# short read means we hit EOF; no more bytes to search
				if len(mini_chunk) < mini_chunk_size:
					boundaries[bi] = file_size
					break
				# step back so a token straddling the boundary isn't missed
				file.seek(start + len(mini_chunk) - overlap)

		return sorted(set(boundaries))
	
	freq_table: dict[tuple[bytes, ...], int] = {}
	num_special = len(special_tokens)

	with open(input_path, "rb") as f:
		boundaries = find_chunk_boundaries(f, 4, b"<|endoftext|>")
		special_pat = "|".join(re.escape(t) for t in special_tokens)
		for start, end in zip(boundaries[:-1], boundaries[1:]):
			f.seek(start)
			chunk = f.read(end - start).decode("utf-8", errors="ignore").replace("\r\n", "\n").replace("\r", "\n")
			for part in re.split(special_pat, chunk):
				for piece, count in pre_tokenize(part).items():
					key = tuple(bytes([b]) for b in piece.encode("utf-8"))
					freq_table[key] = freq_table.get(key, 0) + count

	merges = merge(freq_table, vocab_size, num_special)
	vocab = vocabulary(merges, special_tokens)
	return vocab, merges