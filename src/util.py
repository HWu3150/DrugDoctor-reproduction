from sklearn.metrics import (
    jaccard_score,
    roc_auc_score,
    precision_score,
    f1_score,
    average_precision_score,
)
import json
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import random
import warnings
import dill
from collections import Counter
from collections import defaultdict
try:
    import torch
except ModuleNotFoundError:
    torch = None
try:
    from rdkit import Chem
except ModuleNotFoundError:
    Chem = None

try:
    from ogb.utils import smiles2graph
    from torch_geometric.data import Data
except ModuleNotFoundError:
    smiles2graph = None
    Data = None


warnings.filterwarnings("ignore")


class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)


def load_patient_records_from_jsonl(jsonl_path):
    patient_records = []
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records = obj.get("records") or {}
            diagnoses = records.get("diagnosis") or []
            procedures = records.get("procedure") or []
            medications = records.get("medication") or []

            if not (len(diagnoses) == len(procedures) == len(medications)):
                raise ValueError(
                    f"Line {line_no} has mismatched visit lengths: "
                    f"diag={len(diagnoses)}, proc={len(procedures)}, med={len(medications)}"
                )

            patient = []
            for diag_codes, proc_codes, med_codes in zip(diagnoses, procedures, medications):
                patient.append(
                    [
                        [str(code).strip() for code in diag_codes if str(code).strip()],
                        [str(code).strip() for code in proc_codes if str(code).strip()],
                        [str(code).strip() for code in med_codes if str(code).strip()],
                    ]
                )

            if patient:
                patient_records.append(patient)
    return patient_records


def build_vocab_and_records_from_patient_records(patient_records):
    diag_voc = Voc()
    med_voc = Voc()
    pro_voc = Voc()

    for patient in patient_records:
        for visit in patient:
            diag_voc.add_sentence(visit[0])
            pro_voc.add_sentence(visit[1])
            med_voc.add_sentence(visit[2])

    encoded_records = []
    for patient in patient_records:
        encoded_patient = []
        for visit in patient:
            encoded_patient.append(
                [
                    [diag_voc.word2idx[code] for code in visit[0]],
                    [pro_voc.word2idx[code] for code in visit[1]],
                    [med_voc.word2idx[code] for code in visit[2]],
                ]
            )
        encoded_records.append(encoded_patient)

    voc = {"diag_voc": diag_voc, "pro_voc": pro_voc, "med_voc": med_voc}
    return encoded_records, voc


def load_jsonl_data_and_voc(jsonl_path):
    patient_records = load_patient_records_from_jsonl(jsonl_path)
    return build_vocab_and_records_from_patient_records(patient_records)


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


# use the same metric from DMNC
def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def transform_split(X, Y):
    x_train, x_eval, y_train, y_eval = train_test_split(
        X, Y, train_size=2 / 3, random_state=1203
    )
    x_eval, x_test, y_eval, y_test = train_test_split(
        x_eval, y_eval, test_size=0.5, random_state=1203
    )
    return x_train, x_eval, x_test, y_train, y_eval, y_test


def sequence_output_process(output_logits, filter_token):
    pind = np.argsort(output_logits, axis=-1)[:, ::-1]

    out_list = []
    break_flag = False
    for i in range(len(pind)):
        if break_flag:
            break
        for j in range(pind.shape[1]):
            label = pind[i][j]
            if label in filter_token:
                break_flag = True
                break
            if label not in out_list:
                out_list.append(label)
                break
    y_pred_prob_tmp = []
    for idx, item in enumerate(out_list):
        y_pred_prob_tmp.append(output_logits[idx, item])
    sorted_predict = [
        x for _, x in sorted(zip(y_pred_prob_tmp, out_list), reverse=True)
    ]
    return out_list, sorted_predict


def sequence_metric(y_gt, y_pred, y_prob, y_label):
    def average_prc(y_gt, y_label): 
        target = np.where(y_gt == 1)[0]
        inter = set(y_label) & set(target)
        prc_score = 0 if len(y_label) == 0 else len(inter) / len(y_label)
        return prc_score


    def average_recall(y_gt, y_label):
        target = np.where(y_gt == 1)[0]
        inter = set(y_label) & set(target)
        recall_score = 0 if len(y_label) == 0 else len(inter) / len(target)
        return recall_score


    def average_f1(average_prc, average_recall):
        if (average_prc + average_recall) == 0:
            score = 0
        else:
            score = 2*average_prc*average_recall / (average_prc + average_recall)
        return score


    def jaccard(y_gt, y_label):
        target = np.where(y_gt == 1)[0]
        inter = set(y_label) & set(target)
        union = set(y_label) | set(target)
        jaccard_score = 0 if union == 0 else len(inter) / len(union)
        return jaccard_score

    def f1(y_gt, y_pred):
        all_micro = f1_score(y_gt, y_pred, average='macro')
        return all_micro

    def roc_auc(y_gt, y_pred_prob):
        all_micro = roc_auc_score(y_gt, y_pred_prob, average='macro')
        return all_micro

    def precision_auc(y_gt, y_prob):
        all_micro = average_precision_score(y_gt, y_prob, average='macro')
        return all_micro

    def precision_at_k(y_gt, y_prob_label, k):
        TP = 0
        for j in y_prob_label[:k]:
            if y_gt[j] == 1:
                TP += 1
        precision = TP / k
        return precision 


    try:
        auc = roc_auc(y_gt, y_prob)
    except ValueError:
        auc = 0
    p_1 = precision_at_k(y_gt, y_label, k=1)
    p_3 = precision_at_k(y_gt, y_label, k=3)
    p_5 = precision_at_k(y_gt, y_label, k=5)
    f1 = f1(y_gt, y_pred)
    prauc = precision_auc(y_gt, y_prob)
    ja = jaccard(y_gt, y_label)
    avg_prc = average_prc(y_gt, y_label)
    avg_recall = average_recall(y_gt, y_label)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, prauc, avg_prc, avg_recall, avg_f1


def multi_label_metric(y_gt, y_pred, y_prob):
    def jaccard(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def average_prc(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score

    def average_recall(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score

    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if average_prc[idx] + average_recall[idx] == 0:
                score.append(0)
            else:
                score.append(
                    2
                    * average_prc[idx]
                    * average_recall[idx]
                    / (average_prc[idx] + average_recall[idx])
                )
        return score

    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average="macro"))
        return np.mean(all_micro)

    def roc_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(y_gt[b], y_prob[b], average="macro"))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(
                average_precision_score(y_gt[b], y_prob[b], average="macro")
            )
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob, k=3):
        precision = 0
        sort_index = np.argsort(y_prob, axis=-1)[:, ::-1][:, :k]
        for i in range(len(y_gt)):
            TP = 0
            for j in range(len(sort_index[i])):
                if y_gt[i, sort_index[i, j]] == 1:
                    TP += 1
            precision += TP / len(sort_index[i])
        return precision / len(y_gt)

    # roc_auc
    try:
        auc = roc_auc(y_gt, y_prob)
    except:
        auc = 0
    # precision
    p_1 = precision_at_k(y_gt, y_prob, k=1)
    p_3 = precision_at_k(y_gt, y_prob, k=3)
    p_5 = precision_at_k(y_gt, y_prob, k=5)
    # macro f1
    f1 = f1(y_gt, y_pred)
    # precision
    prauc = precision_auc(y_gt, y_prob)
    # jaccard
    ja = jaccard(y_gt, y_pred)
    # pre, recall, f1
    avg_prc = average_prc(y_gt, y_pred)
    avg_recall = average_recall(y_gt, y_pred)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)


def ddi_rate_score(record, path="../data/output/ddi_A_final.pkl", ddi_adj=None):
    # ddi rate
    ddi_A = ddi_adj if ddi_adj is not None else dill.load(open(path, "rb"))
    all_cnt = 0
    dd_cnt = 0
    for med_code_set in record:
        for i, med_i in enumerate(med_code_set):
            for j, med_j in enumerate(med_code_set):
                if j <= i:
                    continue
                all_cnt += 1
                if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                    dd_cnt += 1
    if all_cnt == 0:
        return 0
    return dd_cnt / all_cnt


def load_ddi_adj_from_atc_csv(csv_path, med_voc):
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        col_codes = [code.strip() for code in header[1:]]

        row_map = {}
        for row in reader:
            row_code = row[0].strip()
            values = row[1:]
            if len(values) != len(col_codes):
                raise ValueError(
                    f"Malformed DDI csv row for {row_code}: "
                    f"{len(values)} values vs {len(col_codes)} columns"
                )
            row_map[row_code] = {
                col_code: float(value) for col_code, value in zip(col_codes, values)
            }

    med_codes = [med_voc.idx2word[i] for i in range(len(med_voc.idx2word))]
    ddi_adj = np.zeros((len(med_codes), len(med_codes)), dtype=np.float32)
    matched_codes = 0

    for i, code_i in enumerate(med_codes):
        row = row_map.get(code_i)
        if row is None:
            continue
        matched_codes += 1
        for j, code_j in enumerate(med_codes):
            value = row.get(code_j)
            if value is not None:
                ddi_adj[i, j] = value

    print(
        f"loaded ddi csv: matched {matched_codes}/{len(med_codes)} med vocab codes "
        f"from {csv_path}"
    )
    return ddi_adj


def create_atoms(mol, atom_dict):
    """Transform the atom types in a molecule (e.g., H, C, and O)
    into the indices (e.g., H=0, C=1, and O=2).
    Note that each atom index considers the aromaticity.
    """
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], "aromatic")
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol, bond_dict):
    """Create a dictionary, in which each key is a node ID
    and each value is the tuples of its neighboring node
    and chemical bond (e.g., single and double) IDs.
    """
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def extract_fingerprints(radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict):
    """Extract the fingerprints from a molecular graph
    based on Weisfeiler-Lehman algorithm.
    """

    if (len(atoms) == 1) or (radius == 0):
        nodes = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges.
            The updated node IDs are the fingerprint IDs.
            """
            nodes_ = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                nodes_.append(fingerprint_dict[fingerprint])

            """Also update each edge ID considering
            its two nodes on both sides.
            """
            i_jedge_dict_ = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    i_jedge_dict_[i].append((j, edge))

            nodes = nodes_
            i_jedge_dict = i_jedge_dict_

    return np.array(nodes)


def buildMPNN(molecule, med_voc, radius=1, device="cpu:0"):
    if Chem is None:
        raise RuntimeError("RDKit is required for molecule processing but is not installed.")
    if torch is None:
        raise RuntimeError("PyTorch is required for molecule processing but is not installed.")

    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))
    MPNNSet, average_index = [], []

    for index, atc3 in med_voc.items():

        smilesList = list(molecule[atc3])
        """Create each data with the above defined functions."""
        counter = 0  # counter how many drugs are under that ATC-3
        for smiles in smilesList:
            try:
                mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                atoms = create_atoms(mol, atom_dict)
                molecular_size = len(atoms)
                i_jbond_dict = create_ijbonddict(mol, bond_dict)
                fingerprints = extract_fingerprints(
                    radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict
                )
                adjacency = Chem.GetAdjacencyMatrix(mol)
                # if fingerprints.shape[0] == adjacency.shape[0]:
                for _ in range(adjacency.shape[0] - fingerprints.shape[0]):
                    fingerprints = np.append(fingerprints, 1)

                fingerprints = torch.LongTensor(fingerprints).to(device)
                adjacency = torch.FloatTensor(adjacency).to(device)
                MPNNSet.append((fingerprints, adjacency, molecular_size))
                counter += 1
            except:
                continue

        average_index.append(counter)

        """Transform the above each data of numpy
        to pytorch tensor on a device (i.e., CPU or GPU).
        """

    N_fingerprint = len(fingerprint_dict)
    # transform into projection matrix
    n_col = sum(average_index)
    n_row = len(average_index)

    average_projection = np.zeros((n_row, n_col))
    col_counter = 0
    for i, item in enumerate(average_index):
        if item > 0:
            average_projection[i, col_counter : col_counter + item] = 1 / item
        col_counter += item

    return MPNNSet, N_fingerprint, torch.FloatTensor(average_projection)



def graph_batch_from_smile(smiles_list):
    if torch is None:
        raise RuntimeError("PyTorch is required for graph construction but is not installed.")
    if smiles2graph is None or Data is None:
        raise RuntimeError(
            "OGB and torch-geometric are required for graph construction but are not installed."
        )
    edge_idxes, edge_feats, node_feats, lstnode, batch = [], [], [], 0, []
    graphs = [smiles2graph(x) for x in smiles_list]
    for idx, graph in enumerate(graphs):
        edge_idxes.append(graph['edge_index'] + lstnode)
        edge_feats.append(graph['edge_feat'])
        node_feats.append(graph['node_feat'])
        lstnode += graph['num_nodes']
        batch.append(np.ones(graph['num_nodes'], dtype=np.int64) * idx)

    result = {
        'edge_index': np.concatenate(edge_idxes, axis=-1),
        'edge_attr': np.concatenate(edge_feats, axis=0),
        'batch': np.concatenate(batch, axis=0),
        'x': np.concatenate(node_feats, axis=0)
    }
    result = {k: torch.from_numpy(v) for k, v in result.items()}
    result['num_nodes'] = lstnode
    return Data(**result)


def patient_to_visit(data,voc_size):
    MED_PAD_TOKEN = voc_size[2] + 2
    diag_list, pro_list, med_list = (
        [],
        [],
        [],
    )  
    # 用于存储全部visit的 diagnose，procedure，medicine 信息。都是list，长度都是visit数，list内部元素是长度不等的对应数据编码
    
    total_visit = 0
    for i in range(len(data)):
        total_visit = total_visit + len(data[i])
    med_emb = np.zeros((total_visit, voc_size[2]))  # array 二维数组，元素是0,1，表示全部visit的真实用药，行是visit数
    patient_visits_num = []  # 记录每个病人的总的visit数，长度是病人数
    visit_tag = []  # 长度最终是visit数，用来标识当前visit是来自哪个病人
    index = 0  # 当前的 visit下标
    for i, input in enumerate(data):
        num = 0
        for idx, adm in enumerate(input):
            diag_list.append(adm[0])
            pro_list.append(adm[1])
            med_list.append(adm[2])
            med_emb[index][adm[2]] = 1
            visit_tag.append(i)
            index = index + 1
            num = num + 1
        patient_visits_num.append(num)

    # e.g.  pro_list:
    # [[0, 1, 2], [3, 4, 1],.....,[16,99]] len(pro_list) == total_visit
    # patient_visits_num: [2, 2, 2, 2, 3, 2, 2, 2, 2, 1, 3, 28, 2, 2, ...]


    ################## 基于索引表的打乱，不直接打乱数据。因为要判定是否可以打乱
    # 为每个病人的visit生成索引列表indexed_arr，其中每个元素是一个元组，元组的第一个元素表示病人，第二个元素表示该病人的visit
    indexed_arr = []
    for patient in range(len(patient_visits_num)):
        for k in range(patient_visits_num[patient]):
            indexed_arr.append((patient, k))
    # [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1), (4, 0), (4, 1), (4, 2), (5, 0), (5, 1), (6, 0), ...]

    
    # 为打乱前的indexed_arr数据创建索引列表
    index_table = list(range(len(indexed_arr)))
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, ...]
    # 打乱顺序
    random.shuffle(index_table)
    # [14052, 4378, 1563, 13103, 5748, 173, 11349, 10676, 7948, 12293, 11167, 11268, 546, 5900, ...]

    # 逐个进行判断，是否遵从打乱？ 只需要保证打乱后的数据的所代表的病人的visit相对顺序不变
    count_list = [0] * len(
        patient_visits_num
    )  # 和 patient_visits_num进行对比，判断新visit是否可以加入当前位置
    final_tuple_list = []

    while len(final_tuple_list) != len(indexed_arr):
        temp_list = []
        for i in range(len(index_table)):
            # index_table[i] == 14052 when i==0
            after_patient_id = indexed_arr[index_table[i]][0]  # 第14052次visit 对应的病人编号5890
            after_patient_visit_id = indexed_arr[index_table[i]][1]  # 对应该病人的visit 1
            # 逻辑判断：
            # 只要当前病人已记录进去的visit数 和当前检索到的这个visit数相等，就说明前面的visit全部录入了
            # 否则，就把这个visit的index先存入temp_list，下一轮再判定
            if after_patient_visit_id == count_list[after_patient_id]:
                final_tuple_list.append(indexed_arr[index_table[i]])
                count_list[after_patient_id] = count_list[after_patient_id] + 1
            else:
                temp_list.append(index_table[i])
        index_table = temp_list  # len(final_tuple_list)+len(temp_list)== 15032 总的visit数

    # print(final_tuple_list)
    # [(5101, 0), (299, 0), (5129, 0), (1889, 0), (5715, 0), (366, 0), (83, 0), (3955, 0), (1498, 0), (1233, 0), (1969, 0), (2398, 0), (975, 0), (4553, 0), ...]
    # 这个tuple是很有用的，比如当训练到第4个visit，假如是 (1889, 2)，
    # 那么可以通过元组检索到这是第1889个病人的第2次visit，那么可以去检索他的第0，1次visit的medicine，作为历史学习

    # 计算出每个病人的第0次visit在总的15032次 visit中的index：visit_index
    # 初始 visit_index: [0, 2, 4, 6, 8, 11, 13, 15, 17, 19, 20, 23, 51, 53, ...]
    visit_index = []
    sum = 0
    for i in range(len(patient_visits_num)):
        visit_index.append(sum)
        sum = sum + patient_visits_num[i]

    # 计算出最终打乱后且“任然有序”的visit索引：visit_index，长度为15032
    # 打乱前：[0,1,2,3,4,5,....,15031]
    # 打乱后：[9975, 6227, 5775, 8899, 241, 7214, 10799,....,..]
    # 目的是，根据打乱后 tuple，知道怎么去把原数据进行打乱（通过final_shuffled_index即可）。
    # 因为tuple只是索引打乱。最终还是转换到数据上。
    final_shuffled_index = []
    for i in range(len(final_tuple_list)):
        final_shuffled_index.append(
            visit_index[final_tuple_list[i][0]] + final_tuple_list[i][1]
        )
    # final_shuffled_index:
    # [9975, 6227, 5775, 8899, 241, 7214, 10799, 7461, 5461, 9158, 476, 5028, 2437, 8077, ...]


    # 利用final_shuffled_index把数据真正打乱！
    shuffled_diag, shuffled_proc, shuffled_med = [], [], []
    # 用于存储 真正打乱后的 visit的 diagnose，procedure，medicine 信息。
    # 都是list，长度都是visit数，list内部元素是长度不等的对应数据编码
    shuffled_med_emb = np.zeros(
        (total_visit, voc_size[2])
    )  # array 二维数组，元素是0,1，表示打乱后的 全部visit的真实用药，行是visit数
    for i in range(total_visit):
        shuffled_diag.append(diag_list[final_shuffled_index[i]])
        shuffled_proc.append(pro_list[final_shuffled_index[i]])
        shuffled_med.append(med_list[final_shuffled_index[i]])
        shuffled_med_emb[i] = med_emb[final_shuffled_index[i]]
    

    # 利用 tuple 得到一个list，表示每个visit的之前几次的用药信息，如果当前visit是该病人的第0次，那么这个list空
    used_med = []
    used_med_emb = [] # 创建0.1标签
    for i in range(len(final_tuple_list)):
        patient_id = final_tuple_list[i][0]#patient_id表示当前visit属于哪个病人
        count = final_tuple_list[i][1] #count表示当前visit前面有几次visit
        if count == 0:
            newList1=[]
            newList2 = []
        else:
            newList1 = []
            newList2 = []
            for j in range(count):
                newList1.append(data[patient_id][j][2])#data[patient_id][j][2]表示这个病人的第j次visit的用药数据
                temp = np.zeros(voc_size[2])
                temp[data[patient_id][j][2]] = 1
                newList2.append(temp)
        used_med.append(newList1)
        used_med_emb.append(newList2)

    # 只要当前visit不是第0次，那么used_med[i]就不会是空，那么就让每个用药信息变成112维，用 MED_PAD_TOKEN 在每次用药记录的最后进行padding，用与后续embedding层编码
    for i in range(len(used_med)):
        count = len(used_med[i])
        if(count!=0):
            for j in range(count):
                used_med[i][j] = used_med[i][j] + [MED_PAD_TOKEN] * (voc_size[2] - len(used_med[i][j]))


    used_diag = []
    for i in range(len(final_tuple_list)):
        patient_id = final_tuple_list[i][0]#patient_id表示当前visit属于哪个病人
        count = final_tuple_list[i][1] #count表示当前visit前面有几次visit
        if count == 0:
            newList=[]
        else:
            newList = []
            for j in range(count):
                newList.append(data[patient_id][j][0])#data[patient_id][j][2]表示这个病人的第j次visit的diag
        used_diag.append(newList)
    
    used_proc = []
    for i in range(len(final_tuple_list)):
        patient_id = final_tuple_list[i][0]#patient_id表示当前visit属于哪个病人
        count = final_tuple_list[i][1] #count表示当前visit前面有几次visit
        if count == 0:
            newList=[]
        else:
            newList = []
            for j in range(count):
                newList.append(data[patient_id][j][1])#data[patient_id][j][2]表示这个病人的第j次visit的proc
        used_proc.append(newList)



    # 为了编码，把每个元素的list信息

    ####################### 上面是处理visit-level的数据，并且打乱，保证各病人自己的visit访问顺序不变
    ####################### 对处理后的visit-level数据进行训练验证测试分割#################
    
    # print("all-patient-size:", len(data))
    # print("all-visit-size:", len(shuffled_med))
    # print(vars(args))
    d_p_m = []
    
    for a, b, c, d, e, f, g, h in zip(shuffled_diag, shuffled_proc, shuffled_med, used_med, used_diag, used_proc, shuffled_med_emb, used_med_emb):
        d_p_m.append([a, b, c, d, e, f, g, h])  
        # 将 shuffled_diag 和 shuffled_proc 和 shuffled_med（当前）相关数据
        # med_true.append(d)  # 将 shuffled_med_emb 作为标签 med_true，长度是112的0,1编码
        # used_med, used_diag, used_proc 表示当前visit的先前visit的数据。
        # 其中 used_med 是长度为112，用token填充了，used_diag 和 used_proc是内部list元素长度不固定的list
    return d_p_m
