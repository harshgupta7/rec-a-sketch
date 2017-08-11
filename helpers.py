import scipy.sparse as sp
import numpy as np


def threshold_interactions_df(df, row_name, col_name, row_min, col_min):
    """Limit interactions df to minimum row and column interactions.

    Parameters
    ----------
    df : DataFrame
        DataFrame which contains a single row for each interaction between
        two entities. Typically, the two entities are a user and an item.
    row_name : str
        Name of column in df which corresponds to the eventual row in the
        interactions matrix.
    col_name : str
        Name of column in df which corresponds to the eventual column in the
        interactions matrix.
    row_min : int
        Minimum number of interactions that the row entity has had with
        distinct column entities.
    col_min : int
        Minimum number of interactions that the column entity has had with
        distinct row entities.
    Returns
    -------
    df : DataFrame
        Thresholded version of the input df. Order of rows is not preserved.

    Examples
    --------

    df looks like:

    user_id | item_id
    =================
      1001  |  2002
      1001  |  2004
      1002  |  2002

    thus, row_name = 'user_id', and col_name = 'item_id'

    If we were to set row_min = 2 and col_min = 1, then the returned df would
    look like

    user_id | item_id
    =================
      1001  |  2002
      1001  |  2004

    """

    n_rows = df[row_name].unique().shape[0]
    n_cols = df[col_name].unique().shape[0]
    sparsity = float(df.shape[0]) / float(n_rows*n_cols) * 100
    print('Starting interactions info')
    print('Number of rows: {}'.format(n_rows))
    print('Number of cols: {}'.format(n_cols))
    print('Sparsity: {:4.3f}%'.format(sparsity))

    done = False
    while not done:
        starting_shape = df.shape[0]
        col_counts = df.groupby(row_name)[col_name].count()
        df = df[~df[row_name].isin(col_counts[col_counts < col_min].index.tolist())]
        row_counts = df.groupby(col_name)[row_name].count()
        df = df[~df[col_name].isin(row_counts[row_counts < row_min].index.tolist())]
        ending_shape = df.shape[0]
        if starting_shape == ending_shape:
            done = True

    n_rows = df[row_name].unique().shape[0]
    n_cols = df[col_name].unique().shape[0]
    sparsity = float(df.shape[0]) / float(n_rows*n_cols) * 100
    print('Ending interactions info')
    print('Number of rows: {}'.format(n_rows))
    print('Number of columns: {}'.format(n_cols))
    print('Sparsity: {:4.3f}%'.format(sparsity))
    return df


def get_df_matrix_mappings(df, row_name, col_name):
    """Map entities in interactions df to row and column indices

    Parameters
    ----------
    df : DataFrame
        Interactions DataFrame.
    row_name : str
        Name of column in df which contains row entities.
    col_name : str
        Name of column in df which contains column entities.

    Returns
    -------
    rid_to_idx : dict
        Maps row ID's to the row index in the eventual interactions matrix.
    idx_to_rid : dict
        Reverse of rid_to_idx. Maps row index to row ID.
    cid_to_idx : dict
        Same as rid_to_idx but for column ID's
    idx_to_cid : dict
    """


    # Create mappings
    rid_to_idx = {}
    idx_to_rid = {}
    for (idx, rid) in enumerate(df[row_name].unique().tolist()):
        rid_to_idx[rid] = idx
        idx_to_rid[idx] = rid

    cid_to_idx = {}
    idx_to_cid = {}
    for (idx, cid) in enumerate(df[col_name].unique().tolist()):
        cid_to_idx[cid] = idx
        idx_to_cid[idx] = cid

    return rid_to_idx, idx_to_rid, cid_to_idx, idx_to_cid


def df_to_matrix(df, row_name, col_name):
    """Take interactions dataframe and convert to a sparse matrix

    Parameters
    ----------
    df : DataFrame
    row_name : str
    col_name : str

    Returns
    -------
    interactions : sparse csr matrix
    rid_to_idx : dict
    idx_to_rid : dict
    cid_to_idx : dict
    idx_to_cid : dict

    """

    rid_to_idx, idx_to_rid,\
        cid_to_idx, idx_to_cid = get_df_matrix_mappings(df,
                                                        row_name,
                                                        col_name)

    def map_ids(row, mapper):
        return mapper[row]

    I = df[row_name].apply(map_ids, args=[rid_to_idx]).as_matrix()
    J = df[col_name].apply(map_ids, args=[cid_to_idx]).as_matrix()
    V = np.ones(I.shape[0])
    interactions = sp.coo_matrix((V, (I, J)), dtype=np.float64)
    interactions = interactions.tocsr()
    return interactions, rid_to_idx, idx_to_rid, cid_to_idx, idx_to_cid


def train_test_split(interactions, split_count=7, fraction=0.2):
    """
    Split recommendation data into train and test sets
    Params
    ------
    interactions : scipy.sparse matrix
        Interactions between users and items.
    split_count : int
        Number of user-item-interactions per user to move
        from training to test set.
    fractions : float
        Fraction of users to split off some of their
        interactions into test set. If None, then all
        users are considered.
    """
    train = interactions.copy().tocoo()
    test = sp.lil_matrix(train.shape)

    select_users = np.where(np.bincount(train.row) >= split_count * 2)[0]
    test_users = np.random.permutation(select_users)[:int((np.floor(fraction*(np.unique(train.row)).shape[0])))]


    if (fraction < 1 and test_users.shape[0] == 0):
        print('Not enough users for {} fraction, setting fraction to 1'.format(fraction))
        test_users = np.where(np.bincount(train.row) >= split_count * 2)[0]
        fraction = 1

    if (test_users.shape[0] == 0 and fraction == 1):
        while(test_users.shape[0] == 0 and split_count >= 2):
            print(('No user with {} interactions, setting split_count to {} and fraction to 1')\
                  .format(2*split_count, split_count - 1))
            split_count -= 1
            test_users = np.where(np.bincount(train.row) >= split_count * 2)[0]

    if test_users.shape[0] ==0:
        print('No user has more than one interaction, cannot split data')
        return None

    train = train.tolil()

    for user in test_users:
        test_interactions = np.random.permutation(interactions.getrow(user).indices)[:((split_count))]
        train[user, test_interactions] = 0.
        test[user, test_interactions] = interactions[user, test_interactions]


    # Test and training are truly disjoint
    assert(train.multiply(test).nnz == 0)
    #All interactions are accounted for 
    assert(interactions.nnz==test.tocsr().nnz+train.tocsr().nnz)
    #Train>=Test
    assert(train.tocsr().nnz>=test.tocsr().nnz)

    return train.tocsr(), test.tocsr()
