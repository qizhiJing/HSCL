import torch

def train_val(num_epochs=400, train_loader=None, val_loader=None, optimizer=None, model=None):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_train = 0
        count_train = 0
        lambda_ = 1 - (epoch / num_epochs) ** (1)
        for data in train_loader:
            mri_, pet_, csf_, label_ = data
            optimizer.zero_grad()
            LOSS, outputs, _ = model(mri=mri_, pet=pet_, csf=csf_, y=label_, lambda_=lambda_)
            LOSS.backward()
            optimizer.step()
            running_loss += LOSS.item()
            _, predicted_train = torch.max(outputs, 1)
            correct_train += (predicted_train == label_).sum().item()
            count_train += len(label_)
        train_accuracy = 100 * correct_train / count_train
        aver_loss = running_loss / len(train_loader)

        model.eval()
        correct = 0
        with torch.no_grad():
            valid_loss = 0.0
            count_val = 0
            for data in val_loader:
                mri_, pet_, csf_, label_ = data
                loss, outputs, loss_ce = model(mri=mri_, pet=pet_, csf=csf_, y=label_, lambda_=lambda_)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == label_).sum().item()
                valid_loss += loss_ce.item()
                count_val += len(label_)
            val_acc = 100 * correct / count_val
            avg_valid_loss = valid_loss / len(val_loader)
            if avg_valid_loss <= best_val_loss:
                best_val_loss = avg_valid_loss
                torch.save(model.state_dict(), 'checkpoint.pt')
                print("-----------------------Model has already been saved-----------------------")

        model.train()
        print('Epoch %d, Loss:%.7f, train_acc:%.2f%%, val_acc:%.2f%%, best_loss:%.2f' %
              (epoch + 1, aver_loss, train_accuracy, val_acc, best_val_loss))
