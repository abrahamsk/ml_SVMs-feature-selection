# list_m = []
#
# for m in xrange(1, 58):
#     print m
#     list_m.append(m)
#
# print "\n"
# print len(list_m)


my_list = [-1.97402703e-02, 1.37206695e-01, 2.66221938e-02, 3.23880858e-02]
sorted_abs_list = sorted(map(abs, my_list))
print sorted_abs_list
import heapq
max_vals = heapq.nlargest(5, sorted_abs_list)
index1 = my_list.index(max_vals[0])
index2 = my_list.index(max_vals[1])
# print index1
# print index2
print max_vals