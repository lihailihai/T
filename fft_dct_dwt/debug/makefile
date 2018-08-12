#=====================================================
#  CUDA NAKEFILE
#  @AUTHOR: HAILI
#  @DATE: 7.10 2018
#     makefile and src in different dir 
#
#====================================================
src_path=..
obj_path=.
src = $(wildcard $(src_path)/*.cu)
obj = $(patsubst %.cu,$(obj_path)/%.o,$(notdir $(src)))
src1 = $(wildcard $(src_path)/*.c)
obj1 = $(patsubst %.c,$(obj_path)/%.o,$(notdir $(src1)))
CC = gcc
CUNV = nvcc
target = result
ALL:$(target)
%.o:$(src_path)/%.cu
	$(CUNV) $< -c -lcusolver -lcusparse -lcublas -lcudart -lcufft $@
%.o:$(src_path)/%.c
	$(CUNV) $< -c $@
$(target):$(obj) $(obj1)
	$(CUNV)  -lcublas -lcusparse -lcudart -lcufft -lcusolver -link -o $@ $(obj) $(obj1)
	@echo 'make file success!'
.PHONY:clean
clean:
	-rm -rf result $(obj) $(obj1)

