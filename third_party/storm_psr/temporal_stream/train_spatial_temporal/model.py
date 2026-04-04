import torch
from torch import nn
import torch.nn.functional as F
from argparse import Namespace
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from safetensors.torch import load_model
from longformer import Longformer
from linformer import Linformer
from transformer import Transformer
from einops.layers.torch import Rearrange
from torchvision import transforms

class VTN(nn.Module):
    """
    From : https://github.com/elb3k/vtn/blob/main/model.py
    """

    def __init__(self, *, frames, num_classes, img_size, patch_size, spatial_frozen, spatial_size, spatial_args, temporal_type, temporal_args, spatial_suffix=''):
        super().__init__()
        self.frames = frames

        # Convert args
        spatial_args = Namespace(**spatial_args)
        temporal_args = Namespace(**temporal_args)

        self.collapse_frames = Rearrange('b f c h w -> (b f) c h w')

        # -- [Spatial] Transformer attention
        self.spatial_transformer = timm.create_model(
            f'vit_{spatial_size}_patch{patch_size}_{img_size}{spatial_suffix}', pretrained=False, **vars(spatial_args))

        # Freeze spatial backbone
        self.spatial_frozen = spatial_frozen
        if spatial_frozen:
            self.spatial_transformer.eval()
        # Spatial preprocess
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(img_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.spatial_transformer.default_cfg['mean'], std=self.spatial_transformer.default_cfg['std'])
        ])
        # Spatial Training preprocess
        config = resolve_data_config({}, model=self.spatial_transformer)
        self.train_preprocess = create_transform(**config, is_training=True)

        # Spatial to temporal rearrange
        self.spatial2temporal = Rearrange('(b f) d -> b f d', f=frames)

        # -- [Temporal] Transformer_attention
        assert temporal_type in ['longformer', 'linformer',
                                 'transformer'], "Only longformer, linformer, transformer are supported"
        # Copy seq_len to frames
        temporal_args.seq_len = frames

        if temporal_type == 'longformer':
            self.temporal_transformer = Longformer(**vars(temporal_args))
        elif temporal_type == 'linformer':
            self.temporal_transformer = Linformer(**vars(temporal_args))
        elif temporal_type == 'transformer':
            self.temporal_transformer = Transformer(**vars(temporal_args))

        # Classifer
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(temporal_args.dim),
            nn.Linear(temporal_args.dim, num_classes)
        )
        # Random init 0.0 mean, 0.02 std
        nn.init.normal_(self.mlp_head[1].weight, mean=0.0, std=0.02)
        self.float()

    def forward(self, img):

        x = self.collapse_frames(img)
        # print(img.size())
        # print(x.size())
        # Spatial Transformer
        if self.spatial_frozen:
            with torch.no_grad():
                x = self.spatial_transformer.forward_features(x)
        else:
            x = self.spatial_transformer.forward_features(x)

        # Spatial to temporal
        x = self.spatial2temporal(x)

        # Temporal Transformer
        x = self.temporal_transformer(x)

        # Classifier
        return self.mlp_head(x)

# -- Temporal feature extractor

class VTN_tmp_only(nn.Module):
    """ Using the pre-extracted embedding (which equavilent to the freeze encoder) to do the temporal modeling.
    From : https://github.com/elb3k/vtn/blob/main/model.py
    """

    def __init__(self, *, frames, num_classes,img_size, spatial_frozen, temporal_type, temporal_args, spatial_suffix=''):
        super().__init__()
        self.frames = frames

        # Convert args
        temporal_args = Namespace(**temporal_args)

        self.collapse_frames = Rearrange('b f c h w -> (b f) c h w')

        # -- [Temporal] Transformer_attention, inpit: b f d
        assert temporal_type in ['longformer', 'linformer',
                                 'transformer'], "Only longformer, linformer, transformer are supported"
        # Copy seq_len to frames
        temporal_args.seq_len = frames

        if temporal_type == 'longformer':
            self.temporal_enc = Longformer(**vars(temporal_args))
        elif temporal_type == 'linformer':
            self.temporal_enc = Linformer(**vars(temporal_args))
        elif temporal_type == 'transformer':
            self.temporal_enc = Transformer(**vars(temporal_args))

        # Classifer
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(temporal_args.dim),
            nn.Linear(temporal_args.dim, num_classes)
        )
        # Random init 0.0 mean, 0.02 std
        nn.init.normal_(self.mlp_head[1].weight, mean=0.0, std=0.02)
        self.float()

    def forward(self, embedding):
        # embedding size: b f d
        # Temporal Transformer
        x = self.temporal_enc(embedding)

        # Classifier
        return self.mlp_head(x)

    def load_weights_encoder(self, path):
        # Load pretrained weigth
        if torch.cuda.is_available():
            self.temporal_enc.load_state_dict(torch.load(path)['temporal_enc'])
            self.mlp_head.load_state_dict(torch.load(path)['mlp_head'])
        else:
            self.temporal_enc.load_state_dict(torch.load(path, map_location="cpu")['temporal_enc'])
            self.mlp_head.load_state_dict(torch.load(path, map_location="cpu")['mlp_head'])

        print(f"Loaded model weights for encoder from {path}")


class No_temporal_encoder(nn.Module):
    """ Using the pre-extracted embedding (which equavilent to the freeze encoder) to do the temporal modeling.
    From : https://github.com/elb3k/vtn/blob/main/model.py
    """

    def __init__(self, frames, num_classes):
        super().__init__()
        #-- NN to replace temporal encoder -> We concatenate frame representations, as means of temporal aggregation
        self.NN = nn.Sequential(
            # The frames * N, N should be calcluated by the input dimension / frame
            nn.Linear(frames*128, 2048)
        )
        # Classifer ->
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(2048),
            nn.Linear(2048, num_classes)
        )

        # Random init 0.0 mean, 0.02 std
        nn.init.normal_(self.mlp_head[1].weight, mean=0.0, std=0.02)
        self.float()

    def forward(self, embedding):
        # embedding size: b f d
        # Reshape input0 [B x Frame x d x 1] to [(B) x (Frame x d) x 1] -> 
        # NN             [B x (Frame x d) x 1] to [B x d x 1]
        # ml_head        [B x d x 1] to [B x 11 x 1]

       
        embedding = embedding.view(embedding.size(0), -1) # From  [B x Frame x d x 1] to [(B) x (Frame x d) x 1]
        #- Aggreation -> We concatenate frame representations, as means of temporal aggregation
        x = self.NN(embedding)

        # Classifier
        return self.mlp_head(x)


class temp_enc_LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, frames, num_classes):
        super().__init__()
        self.lstm = nn.GRU(input_size, hidden_size, frames, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_classes)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
        self.float()

    def forward(self, input_seq):
        # input_Seq -> [Batch X Frame X d x 1]
        output_seq, _ = self.lstm(input_seq)
        # output_seq -> [Batch X Frame X hidden X 1]
        last_frame_output = output_seq[:, -1, :]
        # last_frame ->  [Batch X 1 X hidden X 1]
        pred = self.linear(last_frame_output)
        return pred

# -- Spatial feature extractor
def create_encoder(args, weights_dir=None):
    model_name = args.model
    if "vit" in model_name:
        model = timm.create_model(
            model_name, in_chans=args.channels, num_classes=0)
        strict = False
    else:
        model = timm.create_model(model_name, in_chans=args.channels)
        strict = True
    print(f"Loaded model {model_name}")

    if weights_dir is not None:
        # load_model(
        #     model, f"{weights_dir}.pth", strict=strict)
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(
                weights_dir, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(weights_dir))
        print(f"Loaded pretrained weights from {weights_dir}...")
    else:
        print("Using randomly initialized weights...")

    if "vit" in model_name:
        return model, model.embed_dim
    else:
        layers = torch.nn.Sequential(*list(model.children()))
        try:
            potential_last_layer = layers[-1]
            while not isinstance(potential_last_layer, nn.Linear):
                potential_last_layer = potential_last_layer[-1]
        except TypeError:
            raise TypeError('Can\'t find the linear layer of the model')

        features_dim = potential_last_layer.in_features
        model = torch.nn.Sequential(*list(model.children())[:-1])

        return model, features_dim


class ContrastiveModel(nn.Module):
    """
    adapted from https://github.com/ivanpanshin/SupCon-Framework/blob/main/tools/models.py
    """

    def __init__(self, args, weights_dir=None, classifier=False, classes=23):
        super(ContrastiveModel, self).__init__()
        self.encoder, self.features_dim = create_encoder(args, weights_dir)
        self.projection_head = True
        self.embed_dim = args.hidden
        self.classifier = classifier

        # we don't drop first head
        self.first_head = nn.Sequential(
            nn.Linear(self.features_dim, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.ReLU(inplace=True))

        if self.classifier:
            self.second_head = nn.Sequential(
                nn.Linear(self.embed_dim, classes),
                nn.Softmax())
        else:
            self.second_head = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.BatchNorm1d(self.embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim, args.second_head_input))

    def load_weights_encoder(self, path):
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(path))
        else:
            self.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"Loaded model weights for encoder from {path}")


    def load_weights_encoder_from_ckpt(self, path):
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(path)['spatial_enc'])
        else:
            self.load_state_dict(torch.load(path, map_location="cpu")['spatial_enc'])
        print(f"Loaded model weights for encoder from checkpoint: {path}")

    def use_projection_head(self, mode):
        self.projection_head = mode

    def forward(self, x):
        feat = self.encoder(x)
        feat = self.first_head(feat)
        if self.classifier:
            if self.projection_head:
                logits = self.second_head(feat)
                return logits
            else:
                return F.normalize(feat, dim=0)
        else:
            if self.projection_head:
                return F.normalize(self.second_head(feat), dim=1)
            else:
                if len(feat.shape) == 1:
                    print("Get in to 'if len(feat.shape)'")
                    return F.normalize(feat, dim=0)
                else:
                    return F.normalize(feat, dim=1)


class STORM(nn.Module):
    def __init__(self, *, frames, num_classes, img_size ,spatial_frozen, ckpt_dir, spatial_args,temporal_type, temporal_args,args = None):
        super(STORM, self).__init__()
        # -- Convert args
        self.seq_len = frames
        spatial_args = Namespace(**spatial_args)
        temporal_args = Namespace(**temporal_args)
        # -- [Spatial Encoder]
        #TODO: Add DinoV2-ViTs-backbone here, the output embedding should also be the same (384,1) 
        self.spatial_enc = ContrastiveModel(spatial_args, weights_dir=None, classifier=False)
        if args.spatial_pretrained_weight is None:
            try:
                self.spatial_enc.load_weights_encoder(spatial_args.weights_dir)
                print(f"Done loading pretrained spatial encoder from:{spatial_args.weights_dir}")
            except:
                if "vit" in spatial_args.model:
                    strict = False
                else:
                    strict = True
                load_model(self.spatial_enc, f"{spatial_args.weights_dir}", strict=strict)
                print(f"Loaded pretrained weights from {spatial_args.weights_dir}...")

        else:
            print(f"Done loading pretrained spatial encoder from:{args.spatial_pretrained_weight}")
            self.spatial_enc.load_weights_encoder(args.spatial_pretrained_weight)
                

        self.spatial_enc.use_projection_head(False)  # No need the classificer anymore
        self.collapse_frames = Rearrange('b f c h w -> (b f) c h w')
        self.spatial2temporal = Rearrange('(b f) d -> b f d', f=frames)
        self.permute = Rearrange('f b d -> b f d', f=frames)
        
        # Freeze spatial enc
        self.spatial_frozen = spatial_frozen
        if spatial_frozen:
            print("The spaital encoder is frozen.")
            self.spatial_enc.eval()

        # -- [Temporal Encoder]
         # -- [Temporal] Transformer_attention, inpit: b f d
        assert temporal_type in ['longformer',
                                 'transformer'], "Only longformer, transformer are supported"
        # Copy seq_len to frames
        temporal_args.seq_len = frames
        if temporal_type == 'longformer':
            self.temporal_enc = Longformer(**vars(temporal_args))
        elif temporal_type == 'transformer':
            self.temporal_enc = Transformer(**vars(temporal_args))
        else:
            raise NotImplementedError

        # -- [multi-label classifer]
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(temporal_args.dim),
            nn.Linear(temporal_args.dim, num_classes)
        )
        # Random init 0.0 mean, 0.02 std
        nn.init.normal_(self.mlp_head[1].weight, mean=0.0, std=0.02)
        self.float()

    def forward(self, frames):
        x = self.collapse_frames(frames)
        # -- [Spatial]
        if self.spatial_frozen:
            with torch.no_grad():
                x = self.spatial_enc.forward(x)
        else:
            x = self.spatial_enc.forward(x)
        # -- [Temporal]
        embeddings = self.spatial2temporal(x)
        x = self.temporal_enc(embeddings)
        # -- [Classifier]
        return self.mlp_head(x)

    
class STORM_MLP(nn.Module):
    def __init__(self, *, frames, num_classes, img_size, spatial_frozen, ckpt_dir, spatial_args, temporal_args, args =None):
        super(STORM_MLP, self).__init__()
        spatial_args = Namespace(**spatial_args)
        temporal_args = Namespace(**temporal_args)
        
        # -- Convert args
        self.collapse_frames = Rearrange('b f c h w -> (b f) c h w')
        self.spatial2temporal = Rearrange('(b f) d -> b f d', f=frames)
        
        # -- [Spatial Encoder]
        self.spatial_enc = ContrastiveModel(
            spatial_args, weights_dir=None, classifier=False)
        if args.spatial_pretrained_weight is None:
            self.spatial_enc.load_weights_encoder(args.spatial_pretrained_weight)
        else:
            self.spatial_enc.load_weights_encoder(spatial_args.weights_dir)
        self.spatial_enc.use_projection_head(False)  # No need the classificer anymore

        # Freeze spatial enc
        self.spatial_frozen = spatial_frozen
        if spatial_frozen:
            self.spatial_enc.eval()
        
        # -- [Not using the temporal_enc, but using teh temporal aggregation module]
        self.temporal_enc = No_temporal_encoder(frames,num_classes)
        
        # -- MLP head
        self.mlp_head = No_temporal_encoder(frames,num_classes)
        self.float()

    def forward(self, frames):
        x = self.collapse_frames(frames)
        # -- [Spatial]
        if self.spatial_frozen:
            with torch.no_grad():
                x = self.spatial_enc.forward(x)
        else:
            x = self.spatial_enc.forward(x)

        # -- [Temporal]
        embeddings = self.spatial2temporal(x)
        x = self.mlp_head(embeddings)
        return x
    