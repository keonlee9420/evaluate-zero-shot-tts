import datetime
import os

from yourtts.TTS.utils.io import save_fsspec


def save_checkpoint(model, optimizer, model_loss, out_path, current_step):
    checkpoint_path = "checkpoint_{}.pth.tar".format(current_step)
    checkpoint_path = os.path.join(out_path, checkpoint_path)
    print(" | | > Checkpoint saving : {}".format(checkpoint_path))

    new_state_dict = model.state_dict()
    state = {
        "model": new_state_dict,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": current_step,
        "loss": model_loss,
        "date": datetime.date.today().strftime("%B %d, %Y"),
    }
    save_fsspec(state, checkpoint_path)


def save_best_model(model, optimizer, model_loss, best_loss, out_path, current_step):
    if model_loss < best_loss:
        new_state_dict = model.state_dict()
        state = {
            "model": new_state_dict,
            "optimizer": optimizer.state_dict(),
            "step": current_step,
            "loss": model_loss,
            "date": datetime.date.today().strftime("%B %d, %Y"),
        }
        best_loss = model_loss
        bestmodel_path = "best_model.pth.tar"
        bestmodel_path = os.path.join(out_path, bestmodel_path)
        print("\n > BEST MODEL ({0:.5f}) : {1:}".format(model_loss, bestmodel_path))
        save_fsspec(state, bestmodel_path)
    return best_loss
