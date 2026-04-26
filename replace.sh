#for f in *reg_type[xy].*; do
#  echo "mv $f $(echo $f | sed -E 's/(.*)_bias_(alphasc.*)_reg_type([xy])\.(.*)/\1_bias_\3_\2.\4/')"
#done

for f in *reg_type[xy].*; do
  mv "$f" "$(echo $f | sed -E 's/(.*)_bias_(alphasc.*)_reg_type([xy])\.(.*)/\1_bias_\3_\2.\4/')"
done
