## Examine output from ncdiff and report whether any variables contain non-zero differences.

library(ncdf4)

args = commandArgs(trailingOnly = TRUE)
if(length(args) != 1)
  stop("Usage: Rscript ncdiff_report.R difffile.nc")
  
diffFile = args[1]

print(paste("Checking for differences in", diffFile))

nc = nc_open(diffFile)

vars = names(nc$var)
for(var in vars) {
  
  vals = ncvar_get(nc, var)
  if(!all(vals == 0))
    print(paste("Differences found in variable", var))
}

print("All variables checked.")
nc_close(nc)
