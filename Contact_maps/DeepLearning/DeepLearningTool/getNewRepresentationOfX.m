function newX=getNewRepresentationOfX(nn, X, Y)
newX=[];
m = size(X, 1); %number of instance
n = nn.n; % number of layers
    for i=1:m
        x=X(i, :);
        y=Y(i, :);
        nn = nnff(nn, x, y);
        new_x=getActivationsOfLastHiddenLayer(nn,x,y);
        newX=[newX; new_x];
    end
end
function new_x=getActivationsOfLastHiddenLayer(nn,x,y)
    n = nn.n;
    nn = nnff(nn, x, y);
    new_x=nn.a{n - 1};
    new_x=new_x(2:end);
end
