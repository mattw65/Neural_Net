#include <stdio.h>

int byteSwap(int x, int n, int m) {
int x1 = (0xff << (n << 3)) & x;
int x2 = (0xff << (m << 3)) & x;
x = x ^ x1 ^ x2;
x1 = ((x1 >> (n << 3)) & 0xff) << (m << 3);
x2 = ((x2 >> (m << 3)) & 0xff) << (n << 3);
x = x | x1 | x2;
return !x;

}

int main() {
    
    printf("%d\n", byteSwap(0xfffffffe, 0x0, 0x0));
    
    return 0;
}