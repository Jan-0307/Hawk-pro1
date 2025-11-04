import os, json, hashlib
# ...existing code...
try:
    from web3 import Web3
    HAS_WEB3 = True
except Exception:
    HAS_WEB3 = False

def _sha256_bytes(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()

def compute_root_from_events(events_path="logs/events.log"):
    """Compute a simple Merkle-like root from logs/events.log (one JSON per line).
    Returns hex string root or None if no events.
    """
    if not os.path.exists(events_path):
        return None
    with open(events_path, "rb") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        return None
    nodes = [_sha256_bytes(ln) for ln in lines]
    while len(nodes) > 1:
        next_level = []
        for i in range(0, len(nodes), 2):
            left = nodes[i]
            right = nodes[i+1] if i+1 < len(nodes) else nodes[i]
            next_level.append(_sha256_bytes(left + right))
        nodes = next_level
    return nodes[0].hex()

def push_hash_to_local_log(root_hash, path="logs/blockchain_roots.log"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path,"a", encoding="utf-8") as f:
        f.write(root_hash + "\n")

def record_root_onchain(root_hash):
    """If env vars set and web3 available, send transaction to contract.
    Requires: ETH_RPC, CONTRACT_ADDRESS, PRIVATE_KEY, CONTRACT_ABI_FILE
    Returns tx hash hex on success or None.
    """
    if not HAS_WEB3:
        return None
    ETH_RPC = os.getenv("ETH_RPC")
    CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS")
    PRIVATE_KEY = os.getenv("PRIVATE_KEY")
    CONTRACT_ABI_FILE = os.getenv("CONTRACT_ABI_FILE")
    if not (ETH_RPC and CONTRACT_ADDRESS and PRIVATE_KEY and CONTRACT_ABI_FILE and os.path.exists(CONTRACT_ABI_FILE)):
        return None
    try:
        w3 = Web3(Web3.HTTPProvider(ETH_RPC))
        with open(CONTRACT_ABI_FILE, "r", encoding="utf-8") as af:
            abi = json.load(af)
        contract = w3.eth.contract(address=Web3.to_checksum_address(CONTRACT_ADDRESS), abi=abi)
        acct = w3.eth.account.from_key(PRIVATE_KEY)
        nonce = w3.eth.get_transaction_count(acct.address)
        tx = contract.functions.recordEvent(root_hash).buildTransaction({
            "from": acct.address,
            "nonce": nonce,
            "gas": int(os.getenv("ETHER_GAS", 200000)),
            "gasPrice": int(os.getenv("ETHER_GASPRICE", w3.eth.gas_price))
        })
        signed = acct.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
        return tx_hash.hex()
    except Exception:
        return None

def push_and_optionally_record(root_hash):
    """Write local log and attempt on-chain record. Returns dict with root and tx (or None)."""
    push_hash_to_local_log(root_hash)
    tx = record_root_onchain(root_hash)
    if tx:
        with open("logs/blockchain_roots.log","a", encoding="utf-8") as f:
            f.write("tx:" + tx + "\n")
    return {"root": root_hash, "tx": tx}
# ...existing code...