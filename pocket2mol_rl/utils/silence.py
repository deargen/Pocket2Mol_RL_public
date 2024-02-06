def silence_rdkit():
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
