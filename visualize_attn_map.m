dir = 'attn_stream/';
model = 'self_attn_cyclegan_old_data';
sample = '6';

samples = [53 58 69 72 96 97 100 101];

load([dir model '/self_attn_gen_last_' sample '_fake_B.mat']);

attn_map_combined = attention_scores(1,:);
for i=2:size(attention_scores,1)
    attn_map_combined = attn_map_combined + attention_scores(i,:);
end

attended_regions = attention_scores(:,1);
for i=2:size(attention_scores,2)
    attended_regions = attended_regions + attention_scores(:,i);
end

imshow(reshape(attn_map_combined,[64,64])', 'InitialMagnification','fit')
truesize([224 224]);
%imshow(reshape(attended_regions,[64,64])')
