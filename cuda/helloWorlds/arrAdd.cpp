#include <iostream>

using namespace std;
const int count = 1000000;

void add(int a[], int b[], int count);

int main() {
  int sizeInt = sizeof(int);
  int sizeArr = sizeInt*count;
  int a[count], b[count], c[count]; // host copies of a, b
  for (int i=0;i<count;i++) {
    a[i]=i;
    b[i]=count-i;
  }
  for (int i=0;i<3;i++) {
    cout << a[i] << "   " << b[i] <<endl;
  }
  add(a, b, count);

  cout << "Result is=" << endl;
  for (int i=0;i<3;i++) {
    cout << a[i]<< endl;
  }

  return 0;
}

void add(int a[], int b[], int count) {

  for(int i=0; i<count; i++) {
    a[i] += b[i];
  }
}
