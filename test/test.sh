for file in *
do  
    case "$file" in
    *\.cpp)
        g++ -std=c++14 $file
        ./a.out 1> temp.txt
        diff temp.txt $file.ans
        if [ $? -ne 0 ]; then
            echo "❌$file"
        else
            echo "✅$file"
        fi
        rm temp.txt a.out
    esac
done
