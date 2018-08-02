function [ output ] = choosefunc( lambda, functype, tempH, threshold )

output  = sigmf(tempH,[lambda 0]) - ((functype-1) * (sigmf(tempH,[lambda threshold])));

end

