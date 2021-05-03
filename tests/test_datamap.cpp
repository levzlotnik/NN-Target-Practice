#include "data/data.h"

using namespace data;

int main(int argc, char const *argv[])
{
    auto datamap = DataMap::read_csv("./data.csv");
    
    return 0;
}