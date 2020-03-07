name = registered_files(1).name;
info = h5info(name);
dims = info.Datasets.Dataspace.Size;
ndimsY = length(dims);                       % number of dimensions (data array might be already reshaped)
Ts = dims(end);
for t=1:3
	ObjRecon = read_file(name,t,1);
	MIPs=[max(ObjRecon,[],3) squeeze(max(ObjRecon,[],2));squeeze(max(ObjRecon,[],1))' zeros(size(ObjRecon,3),size(ObjRecon,3))];
	figure;imagesc(MIPs);axis image;
	MIP=uint16(MIPs);
end