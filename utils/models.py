import torch
from torch import nn


class ramen(nn.Module):
    """
    A class for adapting English BERT to other languages
    """

    def __init__(self, src_model, tgt_model, share_position_embs=True):
        """
        src_model: (BertForMaskedLM) of English
        tgt_model: (BertForMaskedLM) of Foreign
        """
        super(ramen, self).__init__()
        self.src_model = src_model
        self.tgt_model = tgt_model
        self.share_position_embs = share_position_embs


        # force sharing params
        self.tgt_model.bert.encoder = self.src_model.bert.encoder
        self.tgt_model.bert.pooler = self.src_model.bert.pooler
        # share embedding params
        if self.share_position_embs:
            self.tgt_model.bert.embeddings.position_embeddings = self.src_model.bert.embeddings.position_embeddings
        # self.tgt_model.bert.embeddings.position_embeddings = self.src_model.bert.embeddings.position_embeddings
        self.tgt_model.bert.embeddings.token_type_embeddings = self.src_model.bert.embeddings.token_type_embeddings
        self.tgt_model.bert.embeddings.LayerNorm = self.src_model.bert.embeddings.LayerNorm
        # share output layers
        self.tgt_model.cls.predictions.transform = self.src_model.cls.predictions.transform

    def forward(self, lang, attention_mask, input_ids, token_type_ids, labels):
        if lang == 'src':
            model = self.src_model
        elif lang == 'tgt':
            model = self.tgt_model
        else:
            raise ValueError("lang should be either 'src' or 'tgt'")
        return model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                     labels=labels)


class AlignAnchor(nn.Module):
    """
    A class for adapting English BERT to other languages
    """

    def __init__(self, src_model, tgt_model, non_shared_src_idx, share_position_embs=True):
        """
        src_model: (BertForMaskedLM) of English
        tgt_model: (BertForMaskedLM) of Foreign
        shared_subw_idx: (list) of the indexes of the shared subwords/embeddings in the src_model
        non_shared_src_idx: (list) of the indexes of the language-specific subwords/embeddings in the src_model
        """
        super(AlignAnchor, self).__init__()
        self.non_shared_src_idx = non_shared_src_idx
        self.share_position_embs = share_position_embs

        # Split the embedding layers of the languages into two (non-consecutive) parts: the language-specific and the shared embeddings
        self.non_shared_tgt_subw = nn.Parameter(
            tgt_model.base_model.embeddings.word_embeddings.weight[self.non_shared_src_idx, :].clone().detach(),
            requires_grad=True)
        # The source embeddings will more or less remain the same
        self.src_embs = nn.Parameter(src_model.base_model.embeddings.word_embeddings.weight.clone().detach(),
                                     requires_grad=True)

        self.src_model = src_model
        self.tgt_model = tgt_model

        del self.src_model.base_model.embeddings.word_embeddings.weight
        del self.tgt_model.base_model.embeddings.word_embeddings.weight
        del self.src_model.cls.predictions.decoder.weight
        del self.tgt_model.cls.predictions.decoder.weight

        torch.cuda.empty_cache()

        # force sharing params
        self.tgt_model.bert.encoder = self.src_model.bert.encoder
        self.tgt_model.bert.pooler = self.src_model.bert.pooler
        # share embedding params
        if self.share_position_embs:
            self.tgt_model.bert.embeddings.position_embeddings = self.src_model.bert.embeddings.position_embeddings
        self.tgt_model.bert.embeddings.token_type_embeddings = self.src_model.bert.embeddings.token_type_embeddings
        self.tgt_model.bert.embeddings.LayerNorm = self.src_model.bert.embeddings.LayerNorm
        # share output layers
        self.tgt_model.cls.predictions.transform = self.src_model.cls.predictions.transform

    def forward(self, lang, attention_mask, input_ids, token_type_ids, labels):
        if lang == 'src':
            # source embeddings are only reset in order for the sharing to work
            self.src_model.base_model.embeddings.word_embeddings.weight = self.src_embs
            model = self.src_model
        elif lang == 'tgt':
            # create new embedding layer for the target language using shared and non-shared embeddings
            self.tgt_model.base_model.embeddings.word_embeddings.weight = self.src_embs.clone()
            # self.tgt_model.base_model.embeddings.word_embeddings.weight[self.non_shared_tgt_idx,
            # :] = self.non_shared_tgt_subw
            # the non shared target embeddings must be positioned in the non-shared source indexes
            self.tgt_model.base_model.embeddings.word_embeddings.weight[self.non_shared_src_idx,
            :] = self.non_shared_tgt_subw
            model = self.tgt_model
        else:
            raise ValueError("lang should be either 'src' or 'tgt'")
        model.tie_weights()
        return model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                     labels=labels)
