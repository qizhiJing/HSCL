import torch
import numpy as np
from util.metrics import compute_metrics


def test(test_loader=None, model=None):
    label_all = []
    all_avg_probs = []
    with torch.no_grad():
        for data in test_loader:
            mri_, pet_, csf_, label_ = data
            label_all.extend(label_.cpu().numpy())
            sum_of_probs = None
            for md in model:
                LOSS, outputs, _ = md(mri=mri_, pet=pet_, csf=csf_, y=label_, lambda_=0)
                probs = torch.softmax(outputs, dim=1)
                if sum_of_probs is None:
                    sum_of_probs = probs
                else:
                    sum_of_probs += probs
            avg_probs = sum_of_probs / len(model)
            all_avg_probs.append(avg_probs.cpu().numpy())

    all_avg_probs = np.vstack(all_avg_probs)
    test_acc, sen, spec, f1, auc = compute_metrics(
        y_true=np.array(label_all),
        y_pro=all_avg_probs)

    print('test: ACC:%.2f%%, Sen:%.2f%%, Spec:%.2f%%, F1:%.2f%%, Auc:%.2f%%' % (test_acc, sen, spec, f1, auc))

    result = {"label": label_all, "pro": avg_probs[:, 1].detach().cpu().numpy(), "acc": test_acc, "sen": sen,
              "spec": spec, "f1": f1, "auc": auc}

    return result