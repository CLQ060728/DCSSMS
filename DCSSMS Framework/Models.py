from config import *


# MLP class for encoder, projector and predictor

def single_MLP_Unit(in_feature_size, out_feature_size):
    return nn.Sequential(
        nn.Linear(in_feature_size, out_feature_size, bias=False),
        nn.BatchNorm1d(out_feature_size),
        nn.ReLU(inplace=True)
    )


def MLP(in_feature_size, out_feature_size, hidden_size=4096):
    return nn.Sequential(
        nn.Linear(in_feature_size, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, out_feature_size)
    )


# my_model = nn.Sequential(nn.Identity(), nn.Identity(), nn.Identity()) print(my_model[0:2])
def SimSiamMLP(in_feature_size, out_feature_size, hidden_size1, hidden_size2):
    return nn.Sequential(
        nn.Linear(in_feature_size, hidden_size1, bias=False),
        nn.BatchNorm1d(hidden_size1),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size1, hidden_size2, bias=False),
        nn.BatchNorm1d(hidden_size2),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size2, out_feature_size, bias=False),
        nn.BatchNorm1d(out_feature_size, affine=False)
    )


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


class Encoder(nn.Module):
    def __init__(self, in_feature_size, num_layers, out_sizes):
        assert in_feature_size > 0, "Input feature size must be greater than 0"
        assert num_layers >= 3, "# layers of the encoder must be greater than or equal to 3"
        assert out_sizes.size(0) == num_layers, "the length of the array of output sizes of the encoder must be equal to the num_layers"
        
        super(Encoder, self).__init__()
        
        self.encoder_unit1 = single_MLP_Unit(in_feature_size, out_sizes[0].item())
        self.encoder_unit2 = single_MLP_Unit(out_sizes[0].item(), out_sizes[1].item())
        self.encoder_unit3 = single_MLP_Unit(out_sizes[1].item(), out_sizes[2].item())
        self.encoder_unit4 = None
        self.encoder_unit5 = None
        self.encoder_unit6 = None
        if num_layers == 4:
            self.encoder_unit4 = single_MLP_Unit(out_sizes[2].item(), out_sizes[3].item())
        if num_layers == 5:
            self.encoder_unit4 = single_MLP_Unit(out_sizes[2].item(), out_sizes[3].item())
            self.encoder_unit5 = single_MLP_Unit(out_sizes[3].item(), out_sizes[4].item())
        if num_layers == 6:
            self.encoder_unit4 = single_MLP_Unit(out_sizes[2].item(), out_sizes[3].item())
            self.encoder_unit5 = single_MLP_Unit(out_sizes[3].item(), out_sizes[4].item())
            self.encoder_unit6 = single_MLP_Unit(out_sizes[4].item(), out_sizes[5].item())
        
        
    def forward(self, x):
        x = self.encoder_unit1(x)
        x = self.encoder_unit2(x)
        x = self.encoder_unit3(x)
        if self.encoder_unit4 is not None:
            x = self.encoder_unit4(x)
        if self.encoder_unit5 is not None:
            x = self.encoder_unit5(x)
        if self.encoder_unit6 is not None:
            x = self.encoder_unit6(x)
        
        return x


class OnlineNetSimCLR(nn.Module):
    def __init__(self, in_feature_size, num_layers, out_sizes, projection_size, projection_hidden_size1, projection_hidden_size2):
        super(OnlineNetSimCLR, self).__init__() 
        self.encoder = Encoder(in_feature_size, num_layers, out_sizes)
        set_requires_grad(self.encoder, True)
        self.online_projector = SimSiamMLP(out_sizes[-1].item(), projection_size, projection_hidden_size1, projection_hidden_size2)
        set_requires_grad(self.online_projector, True)
    
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.online_projector(x)
        
        return x
    
    
    def get_encoder(self):
        assert self.encoder is not None, 'Encoder net has not been created yet'
        return self.encoder
    
    
    def get_projector(self):
        assert self.online_projector is not None, 'Projector net has not been created yet'
        return self.online_projector


class OnlineNetBYOL(nn.Module):
    def __init__(self, in_feature_size, num_layers, out_sizes, projection_size, projection_hidden_size1, projection_hidden_size2,
                 predictor_hidden_size1, predictor_hidden_size2):
        super(OnlineNetBYOL, self).__init__()
        
        self.encoder = Encoder(in_feature_size, num_layers, out_sizes)
        set_requires_grad(self.encoder, True)
        self.online_projector = SimSiamMLP(out_sizes[-1].item(), projection_size, projection_hidden_size1, projection_hidden_size2)
        self.online_predictor = SimSiamMLP(projection_size, projection_size, predictor_hidden_size1, predictor_hidden_size2)
        set_requires_grad(self.online_projector, True)
        set_requires_grad(self.online_predictor, True)
    
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.online_projector(x)
        x = self.online_predictor(x)
        
        return x
    
    
    def get_encoder(self):
        assert self.encoder is not None, 'Encoder net has not been created yet'
        return self.encoder
    
    
    def get_projector(self):
        assert self.online_projector is not None, 'Projector net has not been created yet'
        return self.online_projector


class EMA():
    def __init__(self, tau):
        self.tau = tau
    
    def update_tau(self, new_tau):
        assert new_tau is not None, "Please specify value for new_tau!"
        self.tau = new_tau
    
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.tau + (1 - self.tau) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


class TargetNetBYOL(nn.Module):
    def __init__(self, in_feature_size, num_layers, out_sizes, projection_size, projection_hidden_size1, projection_hidden_size2):
        super(TargetNetBYOL, self).__init__() 
        
        self.encoder = Encoder(in_feature_size, num_layers, out_sizes)
        set_requires_grad(self.encoder, False)
        self.projector = SimSiamMLP(out_sizes[-1].item(), projection_size, projection_hidden_size1, projection_hidden_size2)
        set_requires_grad(self.projector, False)
    
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.projector(x)
        
        return x


class CMDHMSSimCLR(nn.Module):
    def __init__(self, in_feature_size, num_layers, out_sizes, projection_size, projection_hidden_size1 = 4096, projection_hidden_size2 = 4096):
        super(CMDHMSSimCLR, self).__init__()
        
        self.online_net = OnlineNetSimCLR(in_feature_size, num_layers, out_sizes, projection_size, projection_hidden_size1, projection_hidden_size2)
    

    def forward(self, x1, x2):
        z1 = self.online_net(x1)
        z2 = self.online_net(x2)
        
        return z1, z2

        
    def get_online_net(self):
        assert self.online_net is not None, 'Online net has not been created yet'
        return self.online_net
    

    def to_device(self, device):
        assert device is not None, "invalide device value"
        self.online_net.to(device)


class CMDHMSBYOL(nn.Module):
    def __init__(self, in_feature_size, num_layers, out_sizes, projection_size, projection_hidden_size1 = 4096, projection_hidden_size2 = 4096,
                 predictor_hidden_size1 = 4096, predictor_hidden_size2 = 4096, moving_average_decay = 0.996, use_momentum = True):
        super(CMDHMSBYOL, self).__init__()
        
        self.use_momentum = use_momentum
        self.online_net = OnlineNetBYOL(in_feature_size, num_layers, out_sizes, projection_size,
                                        projection_hidden_size1, projection_hidden_size2, predictor_hidden_size1, predictor_hidden_size2)
        self.target_net = TargetNetBYOL(in_feature_size, num_layers, out_sizes, projection_size, projection_hidden_size1, projection_hidden_size2)
        self.target_ema_updater = EMA(moving_average_decay)
    

    def forward(self, x1, x2):
        x1_online = self.online_net(x1)
        x2_online = self.online_net(x2)
        
        if self.use_momentum:
            target_encoder = self.target_net
        else:
            target_encoder = self.online_net
        with T.no_grad():
            x1_target = target_encoder(x1)
            x2_target = target_encoder(x2)
            x1_target.detach_()
            x2_target.detach_()
        
        return x1_online, x1_target, x2_online, x2_target


    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_net is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_net, self.online_net)


    def get_ema_obj(self):
        assert self.target_ema_updater is not None, "EMA object is empty"
        return self.target_ema_updater
        
    def get_online_net(self):
        assert self.online_net is not None, 'Online net has not been created yet'
        return self.online_net
    
    def to_device(self, device):
        assert device is not None, "invalide device value"
        self.online_net.to(device)
        self.target_net.to(device)
    
    
    def get_target_net(self):
        assert self.target_net is not None, 'Target net has not been created yet'
        return self.target_net


class LinearEvalModel(nn.Module):
    def __init__(self, embedding_type, load_path, in_feature_size, num_layers, out_sizes, num_classes, device,
                 projection_size, projection_hidden_size1 = 4096, projection_hidden_size2 = 4096,
                 predictor_hidden_size1 = 4096, predictor_hidden_size2 = 4096):
        super(LinearEvalModel, self).__init__()

        contraNET = None
        best_model_dict = T.load(load_path, map_location=device)
        if embedding_type == "BYOL":
            contraNET = OnlineNetBYOL(in_feature_size, num_layers, out_sizes,
                                      projection_size, projection_hidden_size1, projection_hidden_size2,
                                      predictor_hidden_size1, predictor_hidden_size2)
            contraNET.load_state_dict(best_model_dict["online_model_state_dict"])
        elif embedding_type == "SIMCLR":
            contraNET = OnlineNetSimCLR(in_feature_size, num_layers, out_sizes,
                                        projection_size, projection_hidden_size1, projection_hidden_size2)
            contraNET.load_state_dict(best_model_dict["online_model_state_dict"])
        self.embedding = contraNET.encoder
        set_requires_grad(self.embedding, False)
        self.partial_projector = contraNET.online_projector[0:3]
        set_requires_grad(self.partial_projector, True)
        self.head = nn.Linear(projection_hidden_size2, num_classes, bias=False)
        set_requires_grad(self.head, True)

    def forward(self, x):
        x = self.embedding(x)
        x = self.partial_projector(x)
        x = self.head(x)

        return x


class LinearModel(nn.Module):
    def __init__(self, embedding_type, in_feature_size, num_layers, out_sizes, num_classes,
                 projection_size, projection_hidden_size1 = 4096, projection_hidden_size2 = 4096,
                 predictor_hidden_size1 = 4096, predictor_hidden_size2 = 4096):
        super(LinearModel, self).__init__()

        contraNET = None
        if embedding_type == "BYOL":
            contraNET = OnlineNetBYOL(in_feature_size, num_layers, out_sizes,
                                      projection_size, projection_hidden_size1, projection_hidden_size2,
                                      predictor_hidden_size1, predictor_hidden_size2)
        elif embedding_type == "SIMCLR":
            contraNET = OnlineNetSimCLR(in_feature_size, num_layers, out_sizes,
                                        projection_size, projection_hidden_size1, projection_hidden_size2)
        self.embedding = contraNET.encoder
        set_requires_grad(self.embedding, True)
        self.partial_projector = contraNET.online_projector[0:3]
        set_requires_grad(self.partial_projector, True)
        self.head = nn.Linear(projection_hidden_size2, num_classes, bias=False)
        set_requires_grad(self.head, True)

    def forward(self, x):
        x = self.embedding(x)
        x = self.partial_projector(x)
        x = self.head(x)

        return x


class EmbeddingExtractor(nn.Module):
    def __init__(self, embedding_type, load_path, in_feature_size, num_layers, out_sizes, device,
                 projection_size, projection_hidden_size1 = 4096, projection_hidden_size2 = 4096,
                 predictor_hidden_size1 = 4096, predictor_hidden_size2 = 4096):
        super(EmbeddingExtractor, self).__init__()

        contraNET = None
        best_model_dict = T.load(load_path, map_location=device)
        if embedding_type == "BYOL":
            contraNET = OnlineNetBYOL(in_feature_size, num_layers, out_sizes,
                                      projection_size, projection_hidden_size1, projection_hidden_size2,
                                      predictor_hidden_size1, predictor_hidden_size2)
            contraNET.load_state_dict(best_model_dict["online_model_state_dict"])
        elif embedding_type == "SIMCLR":
            contraNET = OnlineNetSimCLR(in_feature_size, num_layers, out_sizes,
                                        projection_size, projection_hidden_size1, projection_hidden_size2)
            contraNET.load_state_dict(best_model_dict["online_model_state_dict"])
        self.embedding = contraNET.encoder
        set_requires_grad(self.embedding, False)
        self.partial_projector = contraNET.online_projector[0:3]
        set_requires_grad(self.partial_projector, False)

    def forward(self, x):
        x = self.embedding(x)
        x = self.partial_projector(x)

        return x