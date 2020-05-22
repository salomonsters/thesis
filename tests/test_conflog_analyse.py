import pytest
import io

import pandas as pd
import numpy as np

from conflog_analyse import unique_conflicts, inter_and_intra_cluster_conflicts


@pytest.fixture
def conflicts_df():
    conflicts = """simt,ac1,ac2,dcpa,tcpa,tLOS,qdr,dist
    1306.00000000,CKS205__,KLM462__,2436.64433475,296.27088912,111.92728553,348.78758444,8386.65309898
    1306.00000000,KLM462__,CKS205__,2436.64433475,296.27088912,111.92728553,168.78758444,8386.65309898
    1307.00000000,CKS205__,KLM462__,2434.95685171,292.85171167,110.13062199,348.73554398,8366.20650578
    1307.00000000,KLM462__,CKS205__,2434.95685171,292.85171167,110.13062199,168.73554398,8366.20650578
    1308.00000000,CKS205__,KLM462__,2444.66397403,300.87837742,112.59571837,348.68356460,8339.28978834
    1308.00000000,KLM462__,CKS205__,2444.66397403,300.87837742,112.59571837,168.68356460,8339.28978834
    1309.00000000,CKS205__,KLM462__,2454.22326791,297.88439719,111.03948142,348.61436998,8317.17233143
    1309.00000000,KLM462__,CKS205__,2454.22326791,297.88439719,111.03948142,168.61436998,8317.17233143
    1310.00000000,CKS205__,KLM462__,2467.12492828,300.62967034,111.57044877,348.56653658,8291.55000563
    1310.00000000,KLM462__,CKS205__,2467.12492828,300.62967034,111.57044877,168.56653658,8291.55000563
    1312.00000000,CKS205__,KLM462__,2422.23574261,329.76524481,120.57222428,348.50396055,8245.93215868
    1312.00000000,KLM462__,CKS205__,2422.23574261,329.76524481,120.57222428,168.50396055,8245.93215868
    1315.00000000,CKS205__,KLM462__,2365.94802801,333.03868957,119.32111268,348.34554714,8183.22918583
    1315.00000000,KLM462__,CKS205__,2365.94802801,333.03868957,119.32111268,168.34554714,8183.22918583
    1316.00000000,CKS205__,KLM462__,2206.70382914,341.93669648,120.04292285,348.29874930,8161.47678066
    1316.00000000,KLM462__,CKS205__,2206.70382914,341.93669648,120.04292285,168.29874930,8161.47678066
    1316.00000000,CKS204__,KLM162__,2365.94802801,333.03868957,119.32111268,348.34554714,8183.22918583
    1316.00000000,KLM162__,CKS204__,2365.94802801,333.03868957,119.32111268,168.34554714,8183.22918583
    1317.00000000,CKS205__,KLM462__,2031.57322081,351.19887903,120.83074419,348.29998696,8141.18921794
    1317.00000000,KLM462__,CKS205__,2031.57322081,351.19887903,120.83074419,168.29998696,8141.18921794
    1318.00000000,KLM162__,CKS204__,2365.94802801,333.03868957,119.32111268,168.34554714,8183.22918583
    1318.00000000,KLM162__,CKS205__,2031.57322081,351.19887903,120.83074419,168.29998696,8141.18921794
    1318.00000000,CKS204__,KLM162__,2365.94802801,333.03868957,119.32111268,348.34554714,8183.22918583
    1318.00000000,CKS205__,KLM162__,2031.57322081,351.19887903,120.83074419,348.29998696,8141.18921794
    14501.00000000,QTR273__,KLM1766_,273.55029764,374.36419497,195.31178248,3.05067087,11381.23968192
    14501.00000000,KLM1766_,QTR273__,273.55029764,374.36419497,195.31178248,183.05067087,11381.23968192
    14502.00000000,QTR273__,KLM1766_,367.96292396,373.33907293,191.05084868,3.06429103,11360.05611030
    14502.00000000,KLM1766_,QTR273__,367.96292396,373.33907293,191.05084868,183.06429103,11360.05611030
    14503.00000000,QTR273__,KLM1766_,409.17743364,375.18187204,191.64312893,3.04698605,11333.87923780
    14503.00000000,KLM1766_,QTR273__,409.17743364,375.18187204,191.64312893,183.04698605,11333.87923780
    14504.00000000,QTR273__,KLM1766_,567.00445574,373.56052813,190.56233907,3.04343186,11296.67860508
    14504.00000000,KLM1766_,QTR273__,567.00445574,373.56052813,190.56233907,183.04343186,11296.67860508
    14505.00000000,QTR273__,KLM1766_,688.77385964,369.69512737,188.61932952,3.07847479,11277.01138126
    14505.00000000,KLM1766_,QTR273__,688.77385964,369.69512737,188.61932952,183.07847479,11277.01138126
    14506.00000000,QTR273__,KLM1766_,773.20466829,372.06257511,189.61420585,3.07678646,11246.57464779
    14506.00000000,KLM1766_,QTR273__,773.20466829,372.06257511,189.61420585,183.07678646,11246.57464779
    14510.00000000,CKS205__,KLM462__,,,,,
    14510.00000000,KLM462__,CKS205__,,,,,"""
    data = io.StringIO(conflicts)
    df = pd.read_csv(data)
    return df


@pytest.fixture
def conflict_counts_with_cluster_numbers():
    cross_conflicts = {('0483b4fc_48', '0561d8a4_51'): 1}
    conflict_counts_48 = {('c6a497a0_48', 'c8453ae2_48'): 1,
                       ( 'e875bd96_48', 'e948ddde_48'): 1,
                       ('f10c4632_48', 'f54bc42a_48'): 1,
                        ('fdbafaea_48', '1dba7bae_48'): 2,
                       ('200365a6_48', '386fa73a_48'): 2, ('43b42c10_48', '3fa8b172_48'): 1,
                       ('3fa8b172_48', '419ae7d4_48'): 1}
    conflict_counts_51 = {('495bbe1c_51', '4cbde7e2_51'): 1, ('4e45152c_51', '6511a658_51'): 1, ('71a6c57e_51', '7899a59a_51'): 1, ('9b966678_51', 'a0c30bc4_51'): 1,
                       ('b7e72e5c_51', 'dd9d36d2_51'): 2, ('b7e72e5c_51', 'e155a93a_51'): 1,
                       ('f882201c_51', '01582bd2_51'): 1}
    return conflict_counts_48, conflict_counts_51, cross_conflicts


def test_unique_conflicts(conflicts_df):
    with pytest.raises(ValueError):
        unique_conflicts(pd.DataFrame())
    assert unique_conflicts(pd.DataFrame(columns='simt,ac1,ac2,dcpa,tcpa,tLOS,qdr,dist'.split(','))) == dict()

    conflict_counts = unique_conflicts(conflicts_df)
    assert len(conflict_counts) == 4
    assert conflict_counts[("CKS205__", "KLM462__")] == 2
    assert conflict_counts[("CKS204__", "KLM162__")] == 2
    assert conflict_counts[("KLM1766_", "QTR273__")] == 1
    assert conflict_counts[("CKS205__", "KLM162__")] == 1
    _, df_processed = unique_conflicts(conflicts_df, return_df_conflicts=True)
    assert df_processed.shape == (19, 8)

    df_extra_row = pd.concat([conflicts_df.iloc[:4], conflicts_df.iloc[3:7]])
    conflict_counts_extra_row = unique_conflicts(df_extra_row)
    assert conflict_counts_extra_row == {("CKS205__", "KLM462__"): 1}

    df_missing_row = pd.concat([conflicts_df.iloc[:17], conflicts_df.iloc[18:20]])
    conflict_counts_missing_row = unique_conflicts(df_missing_row)
    assert conflict_counts_missing_row == {("CKS205__", "KLM462__"): 1, ("CKS204__", "KLM162__"): 1}

    # Check dt_before_new_conflict
    df_time_difference_0 = conflicts_df.iloc[:4]
    conflict_counts_time_difference_0 = unique_conflicts(df_time_difference_0, dt_before_new_conflict=0)
    assert conflict_counts_time_difference_0 == {("CKS205__", "KLM462__"): 2}

    df_time_difference_1 = conflicts_df.iloc[:16]
    conflict_counts_time_difference_1 = unique_conflicts(df_time_difference_1, dt_before_new_conflict=1)
    assert conflict_counts_time_difference_1 == {("CKS205__", "KLM462__"): 3}

    df_time_difference_2 = conflicts_df.iloc[:16]
    conflict_counts_time_difference_2 = unique_conflicts(df_time_difference_2, dt_before_new_conflict=2)
    assert conflict_counts_time_difference_2 == {("CKS205__", "KLM462__"): 2}

    conflict_minimim_dt_to_count = unique_conflicts(conflicts_df, minimim_duration_for_conflict=1)
    assert len(conflict_minimim_dt_to_count) == 2
    assert conflict_minimim_dt_to_count[("CKS205__", "KLM462__")] == 2
    assert conflict_minimim_dt_to_count[("KLM1766_", "QTR273__")] == 1




def test_inter_and_intra_cluster_conflicts(conflict_counts_with_cluster_numbers):
    c48, c51, c48_51 = conflict_counts_with_cluster_numbers
    assert np.all(inter_and_intra_cluster_conflicts({}) == np.array([[]]))
    matrix_48 = inter_and_intra_cluster_conflicts(c48)
    assert matrix_48[48, 48] == 9
    assert np.sum(matrix_48) == 9
    matrix_51 = inter_and_intra_cluster_conflicts(c51)
    assert matrix_51[51, 51] == 8
    assert np.sum(matrix_51) == 8
    matrix_48_51 = inter_and_intra_cluster_conflicts(c48_51)
    assert matrix_48_51[48, 51] == 1
    assert matrix_48_51[51, 48] == 1
    assert np.sum(matrix_48_51) == 2
    c_combined = {**c48, **c51, **c48_51}
    matrix_combined = inter_and_intra_cluster_conflicts(c_combined)
    assert matrix_combined[48, 48] == 9
    assert matrix_combined[51, 51] == 8
    assert matrix_combined[48, 51] == 1
    assert np.sum(matrix_combined) == 9 + 8 + 2*1