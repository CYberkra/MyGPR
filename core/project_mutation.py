"""One exception boundary for project files, in-memory manifest and catalog."""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import logging
import sqlite3

from core.storage_primitives import FileTransaction


@contextmanager
def project_mutation(store, label, *, files=(), trees=()):
    store.assert_writable()
    before = deepcopy(vars(store.manifest))
    catalog = store.storage.catalog if getattr(store.storage, 'is_hybrid', False) else None
    completed = False
    try:
        with FileTransaction(store.root, label=label) as transaction, store.session.use_file_transaction(transaction):
            if catalog is not None:
                transaction.bind_catalog(catalog.path)
            transaction.track(store.root / 'project.json')
            for path in files:
                transaction.track(path)
            for path in trees:
                transaction.track_tree(path)
            with catalog.transaction() if catalog is not None else nullcontext():
                yield transaction
                if catalog is not None:
                    catalog.set_meta(transaction.catalog_commit_key, 'committed')
                transaction.commit()
        completed = True
    finally:
        if not completed:
            vars(store.manifest).clear()
            vars(store.manifest).update(before)
    if catalog is not None and not transaction.journal_path.exists():
        try:
            with catalog.transaction() as db:
                db.execute('DELETE FROM catalog_meta WHERE key=?', (transaction.catalog_commit_key,))
        except (OSError, sqlite3.Error):
            logging.getLogger(__name__).warning('Deferred file transaction marker cleanup')
